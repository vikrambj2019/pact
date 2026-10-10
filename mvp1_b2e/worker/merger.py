"""Merge per-chunk extraction candidates into a unified draft."""
from __future__ import annotations

import re
import uuid


_STOPWORDS = {"a", "an", "the", "is", "are", "in", "of", "to", "for", "and", "or", "that", "with", "we", "it"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def _similar(a: str, b: str, threshold: float = 0.5) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def _dedup(items: list[dict], key: str, threshold: float = 0.5) -> list[dict]:
    """Remove near-duplicate items by text similarity on the given key field."""
    out: list[dict] = []
    for item in items:
        if not any(_similar(item[key], seen[key], threshold) for seen in out):
            out.append(item)
    return out


def _merge_source_ids(*lists) -> list[str]:
    seen: set[str] = set()
    result = []
    for lst in lists:
        for sid in lst:
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
    return result


def merge_extractions(chunks: list[dict]) -> dict:
    """Merge per-chunk extraction results into one unified draft."""
    all_questions: list[dict] = []
    all_options: list[dict] = []
    all_criteria: list[dict] = []
    all_constraints: list[dict] = []
    all_claims: list[dict] = []
    all_positions: list[dict] = []
    all_objections: list[dict] = []
    all_blockers: list[dict] = []
    all_issues: list[dict] = []
    all_actions: list[dict] = []

    for chunk in chunks:
        all_questions.extend(chunk.get("candidate_questions", []))
        all_options.extend(chunk.get("options", []))
        all_criteria.extend(chunk.get("criteria", []))
        all_constraints.extend(chunk.get("constraints", []))
        all_claims.extend(chunk.get("claims", []))
        all_positions.extend(chunk.get("positions", []))
        all_objections.extend(chunk.get("objections", []))
        all_blockers.extend(chunk.get("blockers", []))
        all_issues.extend(chunk.get("separate_issues", []))
        all_actions.extend(chunk.get("actions", []))

    # Options: merge by title similarity, accumulating source_unit_ids
    merged_options: list[dict] = []
    for opt in all_options:
        existing = next(
            (o for o in merged_options if _similar(opt["title"], o["title"], 0.6)), None
        )
        if existing:
            existing["source_unit_ids"] = _merge_source_ids(
                existing["source_unit_ids"], opt.get("source_unit_ids", [])
            )
            if not existing.get("description") and opt.get("description"):
                existing["description"] = opt["description"]
        else:
            merged_options.append({
                "id": str(uuid.uuid4()),
                "title": opt["title"],
                "description": opt.get("description", ""),
                "source_unit_ids": list(opt.get("source_unit_ids", [])),
                "quote": opt.get("quote", ""),
            })

    # Positions: deduplicate by (participant, option_title, stance) triple
    seen_pos: set[tuple] = set()
    merged_positions: list[dict] = []
    for p in all_positions:
        key = (p["participant_label"].lower(), p["option_title"].lower(), p["stance"])
        if key not in seen_pos:
            seen_pos.add(key)
            merged_positions.append(p)

    return {
        "candidate_questions": _dedup(all_questions, "text", 0.6),
        "options": merged_options,
        "criteria": _dedup(all_criteria, "text"),
        "constraints": _dedup(all_constraints, "text"),
        "claims": _dedup(all_claims, "text", 0.7),
        "positions": merged_positions,
        "objections": _dedup(all_objections, "text"),
        "blockers": _dedup(all_blockers, "text"),
        "separate_issues": _dedup(all_issues, "title"),
        "actions": _dedup(all_actions, "text"),
    }
