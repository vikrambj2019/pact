"""Duplicate detection for source units.

Email chains often re-quote earlier messages verbatim. After normalized_text
strips explicit quote lines (> prefix), residual duplicates are caught here
by token-overlap similarity on the normalized text.
"""
from __future__ import annotations

import re


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def mark_duplicates(units: list[dict]) -> list[dict]:
    """
    Set `_dup_sequence` on each unit dict to the sequence number of the
    earlier unit it duplicates, or None if it is not a duplicate.

    Threshold: overlap / min(len_a, len_b) >= 0.75 and min tokens >= 8.
    Mutates units in-place and returns the same list.
    """
    seen: list[tuple[int, frozenset[str]]] = []
    for unit in units:
        toks = _tokens(unit["normalized_text"])
        dup_seq = None
        if len(toks) >= 8:
            for prev_seq, prev_toks in seen:
                if len(prev_toks) < 8:
                    continue
                overlap = len(toks & prev_toks)
                if overlap / min(len(toks), len(prev_toks)) >= 0.75:
                    dup_seq = prev_seq
                    break
        unit["_dup_sequence"] = dup_seq
        seen.append((unit["sequence"], toks))
    return units
