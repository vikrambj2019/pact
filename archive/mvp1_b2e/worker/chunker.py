"""Split source units into word- and count-bounded chunks for per-chunk LLM extraction."""
from __future__ import annotations

DEFAULT_MAX_WORDS = 2000
# Short chat messages can fit hundreds of units under the word cap. A model reading
# that many tends to extract from the end of the segment and skip the middle, so
# chunks are also capped by message count.
DEFAULT_MAX_UNITS = 30
# Number of trailing units from the previous chunk repeated at the start of the
# next one as read-only context, so a bare reply ("12th") that lands right after
# a chunk boundary can still be read against the question that prompted it.
DEFAULT_OVERLAP_UNITS = 3


def chunk_units(
    units: list[dict],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_units: int = DEFAULT_OVERLAP_UNITS,
    max_units: int | None = DEFAULT_MAX_UNITS,
) -> list[list[dict]]:
    """Group units into chunks of at most ``max_words`` words and ``max_units`` units.

    Each unit stays whole (a single unit longer than ``max_words`` gets its own chunk).
    ``max_units`` of None or 0 disables the count cap.

    Every chunk after the first begins with up to ``overlap_units`` copies of the
    previous chunk's last units, flagged ``context_only=True``. Context units are
    shown to the extractor for reference only; items must not be extracted from
    or cite them. Overlap units do not count toward either cap.
    """
    primary: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for u in units:
        word_count = len(u["normalized_text"].split())
        too_many_words = current_words + word_count > max_words
        too_many_units = bool(max_units) and len(current) >= max_units
        if current and (too_many_words or too_many_units):
            primary.append(current)
            current = []
            current_words = 0
        current.append(u)
        current_words += word_count
    if current:
        primary.append(current)

    if overlap_units <= 0 or len(primary) < 2:
        return primary

    chunks: list[list[dict]] = [primary[0]]
    for prev, chunk in zip(primary, primary[1:]):
        context = [{**u, "context_only": True} for u in prev[-overlap_units:]]
        chunks.append(context + chunk)
    return chunks