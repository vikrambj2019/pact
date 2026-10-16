"""Parse enterprise sources (email chains, Teams transcripts) into the format
normalize.py expects.

Returns a ParsedSource with:
  - transcript   → Transcript (list[RawMessage]) — feed straight into normalize()
  - speakers     → people who actually wrote something (appear in decision matrix)
  - observers    → To/CC people who never authored a message (audience/privacy only)
  - source_type  → "email" | "teams_transcript"
  - subject      → email thread subject or ""
  - msg_meta     → per-message dict: {id, restricted, audience, ...} for privacy tagging

Usage:
    from ingest import parse
    src = parse(Path("ent-08-email-vendor-private-note.txt").read_text())
    spans = normalize(src.transcript, llm)
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Resolve social/ so we can import RawMessage + Transcript
_HERE = Path(__file__).resolve().parent
_SOCIAL = _HERE.parent / "social"
if str(_SOCIAL) not in sys.path:
    sys.path.insert(0, str(_SOCIAL))

from models import RawMessage, Transcript  # noqa: E402


# ---------------------------------------------------------------------------
# Shared text cleanup
# ---------------------------------------------------------------------------

_PRIVATE_USE = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")
_CHAR_FIXES = {
    chr(0xA0): " ", chr(0x202F): " ",
    chr(0x2028): "\n",
    chr(0x2019): "'", chr(0x2018): "'",
    "\r": "",
}


def _clean(text: str) -> str:
    text = (text or "").lstrip(chr(0xFEFF))
    for old, new in _CHAR_FIXES.items():
        text = text.replace(old, new)
    return _PRIVATE_USE.sub("", text)


# ---------------------------------------------------------------------------
# ParsedSource
# ---------------------------------------------------------------------------

@dataclass
class ParsedSource:
    source_type: str                  # "email" | "teams_transcript"
    transcript: Transcript            # feed into normalize()
    speakers: list[str]               # authored ≥1 message, ordered by first appearance
    observers: list[str]              # To/CC only — never spoke
    subject: str = ""                 # email thread subject (or "")
    msg_meta: list[dict] = field(default_factory=list)  # per-message privacy/audience info


# ---------------------------------------------------------------------------
# Source type detection
# ---------------------------------------------------------------------------

_WHATSAPP_LINE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M\s+-\s+[^:]+:", re.MULTILINE)

def detect_source_type(text: str) -> str:
    """Returns 'email', 'teams_transcript', or 'whatsapp'."""
    head = _clean(text)[:6000]
    if len(re.findall(r"(?m)^From: .+", head)) >= 1 and re.search(r"(?m)^(To|Subject|Sent|Date): ", head):
        return "email"
    if re.search(r"(?m)^\d+ (?:hours?|minutes?|seconds?).*\d+:\d\d$", head):
        return "teams_transcript"
    if len(_WHATSAPP_LINE.findall(head)) >= 2:
        return "whatsapp"
    return "teams_transcript"   # fall back to transcript for unknown


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------

_RESTRICT = re.compile(
    r"keep (?:it|this) to yourself|don'?t share|do not share|don'?t forward|do not forward|"
    r"for your eyes only|between us|off the record|not for distribution|don'?t circulate|do not circulate",
    re.I,
)
_HEADER_FIELDS = ("Sent:", "Date:", "To:", "Cc:", "Bcc:", "Subject:")


def _explode_header(match: re.Match) -> str:
    line = match.group(0)
    for f in _HEADER_FIELDS:
        line = re.sub(rf"\s+{re.escape(f)}", f"\n{f}", line, count=1)
    return line


def _people(field: str) -> list[str]:
    out = []
    for m in re.finditer(r"([^;<]+?)\s*<[^>]*>", field or ""):
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
        if name:
            out.append(name)
    if not out:
        out = [p.strip() for p in (field or "").split(";") if p.strip()]
    return list(dict.fromkeys(out))


def _parse_date(s: str) -> datetime | None:
    s = re.sub(r"\s+", " ", (s or "").replace(" at ", " ")).strip()
    s = re.sub(r"^[A-Za-z]+, ", "", s)
    for fmt in ("%B %d, %Y %I:%M %p", "%d %B %Y %H:%M:%S", "%d %B %Y %I:%M %p", "%B %d, %Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _norm_body(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip().lower()


def _parse_email_thread(text: str) -> list[dict]:
    """Parse email thread into unique messages, oldest first."""
    text = re.sub(r"(?m)^From: .*$", _explode_header, _clean(text))
    parts = re.split(r"(?m)^(?=From: )", text)
    cover, blocks = parts[0].strip(), parts[1:]

    raw: list[dict] = []
    for pos, block in enumerate(blocks):
        head, sep, rest = block.partition("\nSubject:")
        if not sep:
            continue
        subject, _, body = rest.partition("\n")
        fields = {}
        for line in head.splitlines():
            k, _, v = line.partition(": ")
            fields[k.strip()] = v.strip()
        when = _parse_date(fields.get("Sent") or fields.get("Date") or "")
        sender = (_people(fields.get("From", "")) or ["Unknown"])[0]
        raw.append({
            "pos": pos, "from": sender, "sent_at": when,
            "subject": subject.strip(),
            "to": _people(fields.get("To", "")),
            "cc": _people(fields.get("Cc", "")),
            "body": body.strip(),
        })

    # Keep only the first occurrence when a message is quoted multiple times.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for m in raw:
        key = (m["from"].lower(),
               m["sent_at"].strftime("%Y%m%d%H%M") if m["sent_at"] else "",
               _norm_body(m["body"])[:300])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    # Audience: running list of everyone who has seen the thread so far (paste order = newest first).
    running: list[str] = []
    restricted_above: str | None = None
    for m in unique:
        own = list(dict.fromkeys(m["to"] + m["cc"]))
        m["audience"] = own
        m["effective_audience"] = list(dict.fromkeys(running + own + [m["from"]]))
        for p in [m["from"]] + own:
            if p not in running:
                running.append(p)
        own_note = _RESTRICT.search(m["body"])
        m["restricted_reason"] = None
        if own_note:
            m["restricted_reason"] = f"'{own_note.group(0)}' ({m['from']})"
            if re.match(r"(?i)(fw|fwd):", m["subject"]):
                restricted_above = restricted_above or m["restricted_reason"]
        elif restricted_above:
            m["restricted_reason"] = restricted_above
        m["restricted"] = bool(m["restricted_reason"])

    # Sort oldest first.
    unique.sort(key=lambda m: (m["sent_at"] or datetime.max, -m["pos"]))

    # Optional cover note (text above the first From: block).
    if cover:
        reason = _RESTRICT.search(cover)
        cover_msg = {
            "pos": -1, "from": "Unknown", "sent_at": None, "subject": "",
            "to": [], "cc": [], "audience": [], "effective_audience": [],
            "body": cover, "audience_unknown": True,
            "restricted": bool(reason),
            "restricted_reason": f"'{reason.group(0)}' (cover note)" if reason else None,
        }
        if reason:
            for m in unique:
                if not m["restricted"]:
                    m["restricted"] = True
                    m["restricted_reason"] = f"'{reason.group(0)}' (cover note)"
        unique.append(cover_msg)

    return unique


def _parse_email(text: str) -> ParsedSource:
    messages = _parse_email_thread(text)

    # Derive subject from the earliest non-cover email.
    subject = next((m["subject"] for m in messages if m.get("subject")), "")
    # Strip RE:/FW: prefixes for the canonical subject.
    subject = re.sub(r"(?i)^(re|fw|fwd):\s*", "", subject).strip()

    # Build RawMessages and track speakers vs observers.
    raw_msgs: list[RawMessage] = []
    all_to_cc: list[str] = []
    speakers_seen: list[str] = []

    for i, m in enumerate(messages, 1):
        author = m["from"]
        ts = m["sent_at"] or datetime.min
        body = m["body"].strip()
        if not body:
            continue
        msg_id = f"m{i}"
        raw_msgs.append(RawMessage(id=msg_id, author=author, ts=ts, text=body))
        if author not in speakers_seen:
            speakers_seen.append(author)
        for p in m.get("to", []) + m.get("cc", []):
            if p not in all_to_cc:
                all_to_cc.append(p)

    observers = [p for p in all_to_cc if p not in speakers_seen]
    msg_meta = [
        {
            "id": f"m{i}",
            "author": m["from"],
            "restricted": m.get("restricted", False),
            "restricted_reason": m.get("restricted_reason"),
            "audience": m.get("effective_audience", []),
            "subject": m.get("subject", ""),
        }
        for i, m in enumerate(messages, 1) if m.get("body", "").strip()
    ]

    return ParsedSource(
        source_type="email",
        transcript=Transcript(channel="email", messages=raw_msgs),
        speakers=speakers_seen,
        observers=observers,
        subject=subject,
        msg_meta=msg_meta,
    )


# ---------------------------------------------------------------------------
# Teams transcript parsing
# ---------------------------------------------------------------------------

_TURN = re.compile(
    r"^(?P<name>[^\d\n][^\n]{1,60}?) (?:(?P<h>\d+) hours? ?)?(?:(?P<m>\d+) minutes? ?)?(?:(?P<s>\d+) seconds?)?$"
)
_GLUED = re.compile(r"^\d+ (?:hours?|minutes?|seconds?)[\w ]*?\d+:\d\d(?::\d\d)?$")
_INITIALS = re.compile(r"^[A-Z]{1,3}$")
_PREAMBLE = ("Transcript. Use arrow keys", "Search", "AI-generated content may be incorrect",
             "Speaker 1 started transcription", "Transcription started", "Transcription stopped")
_GAP_SECONDS = 600


def _turn_header(line: str):
    m = _TURN.match(line)
    if not m or not (m.group("h") or m.group("m") or m.group("s")):
        return None
    secs = int(m.group("h") or 0) * 3600 + int(m.group("m") or 0) * 60 + int(m.group("s") or 0)
    return m.group("name").strip(), secs


def _is_lone_name_block(lines: list[str], i: int) -> bool:
    if i + 2 >= len(lines):
        return False
    name, glued, hdr = lines[i], lines[i + 1], lines[i + 2]
    h = _turn_header(hdr)
    return bool(_GLUED.match(glued)) and h is not None and h[0] == name


def _fmt_ts(secs: int) -> str:
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parse_transcript(text: str) -> ParsedSource:
    lines = [l.strip() for l in _clean(text).split("\n")]
    lines = [l for l in lines if l and not l.startswith(_PREAMBLE)]

    # Strip Teams export decoration (avatar initials, glued time lines, lone name lines).
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _INITIALS.match(line) and _is_lone_name_block(lines, i + 1):
            i += 1
            continue
        if _GLUED.match(line):
            i += 1
            continue
        if _is_lone_name_block(lines, i):
            i += 1
            continue
        cleaned.append(line)
        i += 1

    # Collect turns.
    turns: list[dict] = []
    for line in cleaned:
        hdr = _turn_header(line)
        if hdr:
            turns.append({"speaker": hdr[0], "t": hdr[1], "text": []})
        elif turns:
            turns[-1]["text"].append(line)

    # Merge consecutive same-speaker turns.
    merged: list[dict] = []
    for t in turns:
        body = " ".join(t["text"]).strip()
        if not body:
            continue
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["text"] += " " + body
        else:
            merged.append({"speaker": t["speaker"], "t": t["t"], "text": body})

    # Build RawMessages.
    raw_msgs: list[RawMessage] = []
    speakers_seen: list[str] = []
    epoch = datetime(2000, 1, 1)   # Teams transcripts have relative times only

    for n, t in enumerate(merged, 1):
        ts = datetime.fromtimestamp(epoch.timestamp() + t["t"])
        msg = RawMessage(id=f"m{n}", author=t["speaker"], ts=ts, text=t["text"])
        raw_msgs.append(msg)
        if t["speaker"] not in speakers_seen:
            speakers_seen.append(t["speaker"])

    return ParsedSource(
        source_type="teams_transcript",
        transcript=Transcript(channel="teams", messages=raw_msgs),
        speakers=speakers_seen,
        observers=[],          # no To/CC concept in transcripts
        subject="",
        msg_meta=[{"id": f"m{n}", "author": t["speaker"], "restricted": False,
                   "restricted_reason": None, "audience": [], "subject": ""}
                  for n, t in enumerate(merged, 1)],
    )


# ---------------------------------------------------------------------------
# WhatsApp / generic timestamped chat parsing
# ---------------------------------------------------------------------------

_WA_LINE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+-\s+(?P<name>[^:]+):\s*(?P<body>.*)$"
)


def _parse_whatsapp(text: str) -> ParsedSource:
    lines = _clean(text).splitlines()
    raw_msgs: list[RawMessage] = []
    speakers_seen: list[str] = []
    n = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _WA_LINE.match(line)
        if not m:
            # continuation line — append to last message
            if raw_msgs:
                raw_msgs[-1] = RawMessage(
                    id=raw_msgs[-1].id,
                    author=raw_msgs[-1].author,
                    ts=raw_msgs[-1].ts,
                    text=raw_msgs[-1].text + " " + line,
                )
            continue
        speaker = re.sub(r"\s+", " ", m.group("name")).strip()
        body = m.group("body").strip()
        if not body:
            continue
        try:
            ts = datetime.strptime(f"{m.group('date')} {m.group('time').replace('\u202f', ' ')}", "%m/%d/%y %I:%M %p")
        except ValueError:
            try:
                ts = datetime.strptime(f"{m.group('date')} {m.group('time').replace('\u202f', ' ')}", "%m/%d/%Y %I:%M %p")
            except ValueError:
                ts = datetime.min
        n += 1
        raw_msgs.append(RawMessage(id=f"m{n}", author=speaker, ts=ts, text=body))
        if speaker not in speakers_seen:
            speakers_seen.append(speaker)

    return ParsedSource(
        source_type="whatsapp",
        transcript=Transcript(channel="chat", messages=raw_msgs),
        speakers=speakers_seen,
        observers=[],
        subject="",
        msg_meta=[{"id": f"m{i}", "author": m.author, "restricted": False,
                   "restricted_reason": None, "audience": [], "subject": ""}
                  for i, m in enumerate(raw_msgs, 1)],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(text: str) -> ParsedSource:
    """Auto-detect and parse an email chain, Teams transcript, or WhatsApp chat."""
    source_type = detect_source_type(text)
    if source_type == "email":
        return _parse_email(text)
    if source_type == "whatsapp":
        return _parse_whatsapp(text)
    return _parse_transcript(text)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    path = Path(sys.argv[1])
    src = parse(path.read_text(encoding="utf-8"))
    print(f"source_type : {src.source_type}")
    print(f"subject     : {src.subject!r}")
    print(f"messages    : {len(src.transcript.messages)}")
    print(f"speakers    : {src.speakers}")
    print(f"observers   : {src.observers}")
    print()
    for msg in src.transcript.messages:
        meta = next((m for m in src.msg_meta if m["id"] == msg.id), {})
        flag = " [RESTRICTED]" if meta.get("restricted") else ""
        print(f"  {msg.id}  {msg.author:20}  {msg.ts.strftime('%b %d %H:%M')}  {msg.text[:80]!r}{flag}")
