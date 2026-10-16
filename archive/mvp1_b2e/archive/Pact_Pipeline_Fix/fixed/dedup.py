"""Conservative duplicate detection: repeated views are not duplicate evidence."""
from __future__ import annotations
import re

def mark_duplicates(units: list[dict], enabled: bool = True) -> list[dict]:
    """Only deduplicate exact copies with the same author and known timestamp.

    Token similarity loses corrections, negations and later affirmations. Unknown
    timestamps cannot prove two identical messages represent the same event.
    Chat/transcript callers disable this: separate turns remain separate evidence.
    """
    seen = {}
    for unit in units:
        text = re.sub(r"\s+", " ", unit["normalized_text"]).strip()
        timestamp = unit.get("timestamp")
        key = (unit.get("author_label"), timestamp, text)
        unit["_dup_sequence"] = seen.get(key) if enabled and timestamp is not None else None
        if enabled and timestamp is not None:
            seen.setdefault(key, unit["sequence"])
    return units
