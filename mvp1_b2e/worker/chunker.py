"""Split source units into word-bounded chunks for per-chunk LLM extraction."""
from __future__ import annotations

DEFAULT_MAX_WORDS = 2000


def chunk_units(units: list[dict], max_words: int = DEFAULT_MAX_WORDS) -> list[list[dict]]:
    """Group units into chunks not exceeding max_words. Each unit stays whole."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for u in units:
        word_count = len(u["normalized_text"].split())
        if current and current_words + word_count > max_words:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(u)
        current_words += word_count
    if current:
        chunks.append(current)
    return chunks
