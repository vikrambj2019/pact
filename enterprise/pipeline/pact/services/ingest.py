"""Turn pasted email threads and Teams transcript exports into source units (step 1, before chunking).

Both parsers return plain dicts shaped like the rows the pipeline already stores::

    {"sequence": int, "author_label": str, "normalized_text": str, "locator": {...}}

The caller creates ``SourceUnit`` rows from them; nothing else in the pipeline changes.
Email audience metadata and the transcript time stamps live in ``locator`` (see audience.py).
"""
from __future__ import annotations

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Written with chr() so the source stays plain ASCII.
_PRIVATE_USE = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")      # icon glyphs in Teams exports
_CHAR_FIXES = {
    chr(0xA0): " ", chr(0x202F): " ",              # non-breaking spaces
    chr(0x2028): "\n",                              # Outlook line separator inside headers
    chr(0x2019): "'", chr(0x2018): "'",             # curly apostrophes
    "\r": "",
}


def _clean(text: str) -> str:
    text = (text or "").lstrip(chr(0xFEFF))
    for old, new in _CHAR_FIXES.items():
        text = text.replace(old, new)
    return _PRIVATE_USE.sub("", text)


def detect_source_type(text: str) -> str:
    """'email' | 'teams_transcript' | 'chat'. Cheap heuristics on the first part of the text."""
    head = _clean(text)[:6000]
    if len(re.findall(r"(?m)^From: .+", head)) >= 1 and re.search(r"(?m)^(To|Subject|Sent|Date): ", head):
        return "email"
    if re.search(r"(?m)^\d+ (?:hours?|minutes?|seconds?).*\d+:\d\d$", head):
        return "teams_transcript"
    return "chat"


# ---------------------------------------------------------------------------
# Email threads
# ---------------------------------------------------------------------------

_RESTRICT = re.compile(
    r"keep (?:it|this) to yourself|don'?t share|do not share|don'?t forward|do not forward|"
    r"for your eyes only|between us|off the record|not for distribution|don'?t circulate|do not circulate",
    re.I,
)
_HEADER_FIELDS = ("Sent:", "Date:", "To:", "Cc:", "Bcc:", "Subject:")


def _explode_header(match: re.Match) -> str:
    """Some exports flatten a header onto one line; put each field on its own line."""
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
    if not out:  # addresses without display names
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


def parse_email_thread(text: str) -> list[dict]:
    """Split a pasted thread (newest first, each reply quoting the older mail) into unique messages.

    * one message per header block; a message quoted again below another is kept once
    * messages come back oldest first, plus a cover note (text above the first header) last
    * ``effective_audience`` = recipients + sender + everyone on any message above it in the paste,
      because those messages carried it as quoted history
    * a privacy request ("keep it to yourself", "don't share") marks that message as restricted; in a forward or
      cover note it also covers everything quoted below it
    """
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
            "pos": pos, "from": sender, "sent_at": when, "subject": subject.strip(),
            "to": _people(fields.get("To", "")), "cc": _people(fields.get("Cc", "")),
            "body": body.strip(),
        })

    # The same message appears once per reply that quotes it: keep the first copy.
    seen_keys: set[tuple] = set()
    unique: list[dict] = []
    for m in raw:
        key = (m["from"].lower(), m["sent_at"].strftime("%Y%m%d%H%M") if m["sent_at"] else "", _norm_body(m["body"])[:300])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(m)

    # Audience bookkeeping runs in paste order (newest first): a message is quoted by everything above it.
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
            # A plain reply's request covers only that reply. A forward asks about the whole pile it carries.
            if re.match(r"(?i)(fw|fwd):", m["subject"]):
                restricted_above = restricted_above or m["restricted_reason"]
        elif restricted_above:
            m["restricted_reason"] = restricted_above
        m["restricted"] = bool(m["restricted_reason"])

    unique.sort(key=lambda m: (m["sent_at"] or datetime.max, -m["pos"]))

    out = list(unique)
    if cover:
        reason = _RESTRICT.search(cover)
        out.append({
            "pos": -1, "from": "Unknown", "sent_at": None, "subject": "", "to": [], "cc": [], "audience": [],
            "effective_audience": [], "body": cover, "audience_unknown": True,
            "restricted": bool(reason), "restricted_reason": f"'{reason.group(0)}' (cover note)" if reason else None,
        })
        if reason:  # a cover note asking for privacy covers everything pasted under it
            for m in out:
                if not m["restricted"]:
                    m["restricted"], m["restricted_reason"] = True, f"'{reason.group(0)}' (cover note)"
    return out


def email_units(text: str) -> list[dict]:
    units = []
    for i, m in enumerate(parse_email_thread(text), 1):
        subject = m["subject"]
        kind = "forward" if re.match(r"(?i)(fw|fwd):", subject) else ("reply" if re.match(r"(?i)re:", subject) else "new")
        if m.get("audience_unknown"):
            kind = "cover_note"
        sent = m["sent_at"]
        units.append({
            "sequence": i,
            "author_label": m["from"],
            "normalized_text": m["body"],
            "locator": {
                "source_type": "email",
                "timestamp_str": sent.strftime("%b %d, %I:%M %p").replace(" 0", " ") if sent else "",
                "sent_at": sent.isoformat(timespec="minutes") if sent else None,
                "subject": subject, "msg_kind": kind,
                "to": m["to"], "cc": m["cc"], "audience": m["audience"],
                "effective_audience": m["effective_audience"],
                "audience_unknown": bool(m.get("audience_unknown")),
                "restricted": m["restricted"], "restricted_reason": m["restricted_reason"],
            },
        })
    return units


# ---------------------------------------------------------------------------
# Teams transcript exports
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


def _fmt_ts(secs: int) -> str:
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_teams_transcript(text: str) -> list[dict]:
    """Parse a Teams 'Transcript' copy/paste into one unit per speaker turn.

    The export repeats each speaker three ways (name alone, a glued time line, then
    'Name N minutes M seconds'); only the last form starts a turn. Consecutive blocks by the
    same speaker are merged. Each unit records its time and any gap since the previous unit, so
    an excerpt that jumps from minute 1 to minute 63 is visible to the extractor.
    """
    lines = [l.strip() for l in _clean(text).split("\n")]
    lines = [l for l in lines if l and not l.startswith(_PREAMBLE)]

    # Drop the decoration around each speaker block.
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _INITIALS.match(line) and _is_lone_name_block(lines, i + 1):
            i += 1                                  # avatar initials ("BV") before a speaker block
            continue
        if _GLUED.match(line):
            i += 1
            continue
        if _is_lone_name_block(lines, i):
            i += 1                                  # name alone, repeated in the header that follows
            continue
        cleaned.append(line)
        i += 1

    turns: list[dict] = []
    for line in cleaned:
        hdr = _turn_header(line)
        if hdr:
            turns.append({"speaker": hdr[0], "t": hdr[1], "text": []})
        elif turns:
            turns[-1]["text"].append(line)

    merged: list[dict] = []
    for t in turns:
        body = " ".join(t["text"]).strip()
        if not body:
            continue
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["text"] += " " + body
        else:
            merged.append({"speaker": t["speaker"], "t": t["t"], "text": body})

    units, prev_t = [], None
    for n, t in enumerate(merged, 1):
        loc = {"source_type": "teams_transcript", "timestamp_str": _fmt_ts(t["t"]), "t_seconds": t["t"]}
        if prev_t is not None and t["t"] - prev_t > _GAP_SECONDS:
            loc["gap_before_s"] = t["t"] - prev_t
        prev_t = t["t"]
        units.append({"sequence": n, "author_label": t["speaker"], "normalized_text": t["text"], "locator": loc})
    return units


def _is_lone_name_block(lines: list[str], i: int) -> bool:
    """True if lines[i] is a bare speaker name followed by a glued time line and then 'Name <time>'."""
    if i + 2 >= len(lines):
        return False
    name, glued, hdr = lines[i], lines[i + 1], lines[i + 2]
    h = _turn_header(hdr)
    return bool(_GLUED.match(glued)) and h is not None and h[0] == name
