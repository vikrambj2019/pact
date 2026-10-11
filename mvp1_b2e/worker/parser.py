"""Source document parser — Step 2 (Ingestion).

Supports: email_chain (RFC-822 style, --- delimited), transcript ([MM:SS] format), plain_text.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..api.config import settings
from ..api.models import SourceDocument, SourceUnit, ParseStatus
from ..api.storage import LocalStorage


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_WHATSAPP_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M\s+-\s+.+:",
    re.MULTILINE,
)


def _detect_format(text: str, declared_type: str) -> str:
    if _TEAMS_MSG_RE.search(text):
        return "teams"
    if declared_type == "email_chain":
        return "email_chain"
    if declared_type == "transcript":
        return "transcript"
    if re.search(r"^From:\s+.+\n", text, re.MULTILINE) and re.search(r"^Message-ID:", text, re.MULTILINE):
        return "email_chain"
    if re.search(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]", text, re.MULTILINE):
        return "transcript"
    if _WHATSAPP_RE.search(text):
        return "whatsapp"
    return "plain_text"


# ---------------------------------------------------------------------------
# Email chain parser
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(From|To|Date|Subject|Message-ID|In-Reply-To|CC):\s*(.+)$", re.IGNORECASE)
_FROM_NAME_RE = re.compile(r"^(.+?)\s*<[^>]+>$")
_QUOTE_HEADER_RE = re.compile(r"^On .+ wrote:$")


def _strip_quoted(body_lines: list[str]) -> str:
    out = []
    for line in body_lines:
        s = line.strip()
        if s.startswith(">") or _QUOTE_HEADER_RE.match(s):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _parse_email_chain(text: str) -> list[dict]:
    blocks = re.split(r"\n---+\n", text)
    units = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        headers: dict[str, str] = {}
        body_start = len(lines)
        for j, line in enumerate(lines):
            m = _HEADER_RE.match(line)
            if m:
                headers[m.group(1).lower()] = m.group(2).strip()
            elif line.strip() == "" and headers:
                body_start = j + 1
                break
        body_lines = lines[body_start:]
        original_text = "\n".join(body_lines).strip()
        normalized_text = _strip_quoted(body_lines) or original_text

        author_label: Optional[str] = None
        from_val = headers.get("from", "")
        if from_val:
            nm = _FROM_NAME_RE.match(from_val)
            author_label = nm.group(1).strip() if nm else from_val.split("<")[0].strip() or from_val

        timestamp: Optional[datetime] = None
        try:
            timestamp = parsedate_to_datetime(headers["date"])
        except Exception:
            pass

        units.append({
            "sequence": i,
            "author_label": author_label,
            "timestamp": timestamp,
            "original_text": original_text,
            "normalized_text": normalized_text,
            "locator": {
                "sequence": i,
                "message_id": headers.get("message-id"),
                "in_reply_to": headers.get("in-reply-to"),
                "subject": headers.get("subject"),
            },
        })
    return units


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------

_TURN_RE = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+(.+?):\s+(.+)$")


def _ts_offset(ts_str: str) -> Optional[datetime]:
    try:
        parts = [int(x) for x in ts_str.split(":")]
        delta = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]) if len(parts) == 3 \
            else timedelta(minutes=parts[0], seconds=parts[1])
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + delta
    except Exception:
        return None


def _parse_transcript(text: str) -> list[dict]:
    units = []
    seq = 0
    for line in text.splitlines():
        m = _TURN_RE.match(line.strip())
        if not m:
            if units and line.strip():
                units[-1]["original_text"] += "\n" + line.strip()
                units[-1]["normalized_text"] += "\n" + line.strip()
            continue
        ts_str, speaker, content = m.group(1), m.group(2).strip(), m.group(3).strip()
        units.append({
            "sequence": seq,
            "author_label": speaker,
            "timestamp": _ts_offset(ts_str),
            "original_text": content,
            "normalized_text": content,
            "locator": {"sequence": seq, "timestamp_str": ts_str},
        })
        seq += 1
    return units


# ---------------------------------------------------------------------------
# WhatsApp export parser
# ---------------------------------------------------------------------------

_WA_MSG_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}\s+[AP]M)\s+-\s+([^:]+):\s+(.+)$"
)
_WA_SKIP_RE = re.compile(
    r"^(<Media omitted>|This message was deleted|Messages and calls are end-to-end encrypted.*)$",
    re.IGNORECASE,
)


def _parse_whatsapp(text: str) -> list[dict]:
    units = []
    seq = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _WA_MSG_RE.match(line)
        if not m:
            if units and line and not re.match(r"^\d{1,2}/\d{1,2}/\d{2,4},", line):
                units[-1]["original_text"] += "\n" + line
                units[-1]["normalized_text"] += "\n" + line
            continue
        date_str, time_str, author, content = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        if _WA_SKIP_RE.match(content):
            continue
        # Preserve links as source evidence; their presence is not verification.
        content_clean = content
        if not content_clean:
            continue
        timestamp: Optional[datetime] = None
        try:
            timestamp = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %I:%M %p").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %I:%M %p").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        ts_str = f"{date_str} {time_str}"
        units.append({
            "sequence": seq,
            "author_label": author,
            "timestamp": timestamp,
            "original_text": content_clean,
            "normalized_text": content_clean,
            "locator": {"sequence": seq, "timestamp_str": ts_str},
        })
        seq += 1
    return units


# ---------------------------------------------------------------------------
# Plain text parser
# ---------------------------------------------------------------------------

def _parse_plain_text(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return [
        {
            "sequence": i,
            "author_label": None,
            "timestamp": None,
            "original_text": p,
            "normalized_text": p,
            "locator": {"sequence": i},
        }
        for i, p in enumerate(paragraphs)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def parse_source(payload: dict, session: AsyncSession) -> dict:
    from .dedup import mark_duplicates
    from .participants import upsert_participants

    source_id = uuid.UUID(payload["source_document_id"])
    doc: SourceDocument = await session.get(SourceDocument, source_id)
    if doc is None:
        raise ValueError(f"SourceDocument {source_id} not found")

    doc.parse_status = ParseStatus.processing
    await session.commit()

    storage = LocalStorage(Path(settings.storage_local_path))
    data = await storage.get(doc.storage_key)
    text = data.decode("utf-8", errors="replace")

    fmt = _detect_format(text, doc.type)
    warnings: list[str] = []

    if fmt == "teams":
        raw_units = _parse_teams(text)
    elif fmt == "email_chain":
        raw_units = _parse_email_chain(text)
    elif fmt == "transcript":
        raw_units = _parse_transcript(text)
    elif fmt == "whatsapp":
        raw_units = _parse_whatsapp(text)
    else:
        raw_units = _parse_plain_text(text)

    if not raw_units:
        warnings.append("No content units found in document")

    raw_units = mark_duplicates(raw_units, enabled=fmt == "email_chain")

    unique_labels = list(dict.fromkeys(u["author_label"] for u in raw_units if u.get("author_label")))
    participant_map = await upsert_participants(doc.workspace_id, unique_labels, session)

    seq_to_unit_id: dict[int, uuid.UUID] = {}
    unit_count = 0
    for u in raw_units:
        dup_seq = u.pop("_dup_sequence", None)
        unit = SourceUnit(
            document_id=doc.id,
            workspace_id=doc.workspace_id,
            sequence=u["sequence"],
            author_label=u["author_label"],
            mapped_participant_id=participant_map.get(u["author_label"]) if u.get("author_label") else None,
            timestamp=u["timestamp"],
            original_text=u["original_text"],
            normalized_text=u["normalized_text"],
            locator=u["locator"],
            duplicate_of=seq_to_unit_id.get(dup_seq) if dup_seq is not None else None,
        )
        session.add(unit)
        await session.flush()
        seq_to_unit_id[u["sequence"]] = unit.id
        unit_count += 1

    doc.parse_status = ParseStatus.done
    doc.warnings = warnings
    await session.commit()

    return {
        "source_document_id": str(doc.id),
        "format": fmt,
        "unit_count": unit_count,
        "participant_count": len(participant_map),
        "warnings": warnings,
    }
# Teams text export (wall-clock times have no timezone in the source).
_TEAMS_MSG_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}\s+[AP]M)\]\s+([^:\n]+):\s*(.*)$",
    re.MULTILINE,
)


def _parse_teams(text: str) -> list[dict]:
    units = []
    for line in text.splitlines():
        match = _TEAMS_MSG_RE.fullmatch(line.strip())
        if not match:
            if units and line.strip():
                units[-1]["original_text"] += "\n" + line.strip()
                units[-1]["normalized_text"] += "\n" + line.strip()
            continue
        date, time, author, content = match.groups()
        ts = None
        for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
            try:
                ts = datetime.strptime(date + " " + time, fmt)
                break
            except ValueError:
                pass
        seq = len(units)
        units.append({"sequence": seq, "author_label": author.strip(), "timestamp": ts,
                      "original_text": content, "normalized_text": content,
                      "locator": {"sequence": seq, "timestamp_str": date + " " + time,
                                  "timezone_unspecified": True}})
    return units

