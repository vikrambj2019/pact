"""Per-chunk LLM extraction of decision brief candidates."""
from __future__ import annotations

_SYSTEM = """\
You are a precise extraction assistant for decision analysis. Read the discussion segment below and extract structured elements relevant to the decision being made.

Hard rules:
- Only extract what is explicitly stated. Do not infer, generalize, or fabricate.
- For every extracted item, provide the exact verbatim quote from the text.
- A claim is what a participant asserted — it is NOT independently verified fact.
- Silence or lack of objection NEVER implies agreement or a position.
- Participant positions require explicit statements (e.g. "I support X", "I'm opposed to Y").
- If a topic appears tangential to the main decision, flag it as a separate_issue.
- Use the unit IDs provided in [unit:...] markers for source_unit_ids.
- Leave lists empty ([]) if nothing of that type appears in this segment.
"""

_EXTRACT_TOOL = {
    "name": "extract_candidates",
    "description": "Extract decision-relevant candidates from a discussion segment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_questions": {
                "type": "array",
                "description": "Decision questions stated or implied in this segment.",
                "items": {
                    "type": "object",
                    "required": ["text", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "options": {
                "type": "array",
                "description": "Specific options or alternatives being considered.",
                "items": {
                    "type": "object",
                    "required": ["temp_id", "title", "source_unit_ids", "quote"],
                    "properties": {
                        "temp_id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "criteria": {
                "type": "array",
                "description": "Evaluation criteria or factors that should guide the decision.",
                "items": {
                    "type": "object",
                    "required": ["text", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "constraints": {
                "type": "array",
                "description": "Hard or soft constraints on what decisions are acceptable.",
                "items": {
                    "type": "object",
                    "required": ["text", "kind", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string", "enum": ["mandatory", "negotiable", "unknown"]},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "claims": {
                "type": "array",
                "description": "Factual, predictive, or causal assertions made by participants.",
                "items": {
                    "type": "object",
                    "required": ["text", "kind", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string", "enum": ["factual", "predictive", "causal", "priority", "constraint", "other"]},
                        "author_label": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "positions": {
                "type": "array",
                "description": "Explicit participant stances on specific options. Only include if the participant explicitly stated their stance.",
                "items": {
                    "type": "object",
                    "required": ["participant_label", "option_title", "stance", "source_unit_ids", "quote"],
                    "properties": {
                        "participant_label": {"type": "string"},
                        "option_title": {"type": "string"},
                        "stance": {"type": "string", "enum": ["support", "oppose", "conditional", "undecided"]},
                        "conditions": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "objections": {
                "type": "array",
                "description": "Explicit concerns or objections raised against an option or direction.",
                "items": {
                    "type": "object",
                    "required": ["text", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "option_titles": {"type": "array", "items": {"type": "string"}},
                        "author_label": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "blockers": {
                "type": "array",
                "description": "Unresolved questions, missing information, or dependencies that block the decision.",
                "items": {
                    "type": "object",
                    "required": ["text", "kind", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string", "enum": ["open_question", "unresolved_dependency", "missing_info"]},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "separate_issues": {
                "type": "array",
                "description": "Topics that appear tangential or unrelated to the main decision thread.",
                "items": {
                    "type": "object",
                    "required": ["title", "rationale", "source_unit_ids"],
                    "properties": {
                        "title": {"type": "string"},
                        "rationale": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "actions": {
                "type": "array",
                "description": "Explicitly assigned tasks or follow-up actions.",
                "items": {
                    "type": "object",
                    "required": ["text", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "owner_label": {"type": "string"},
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            }
        },
        "required": ["candidate_questions", "options", "criteria", "constraints", "claims",
                     "positions", "objections", "blockers", "separate_issues", "actions"]
    }
}


def _format_units(units: list[dict]) -> str:
    lines = []
    for u in units:
        ts = f"[{u['locator'].get('timestamp_str', '')}] " if u["locator"].get("timestamp_str") else ""
        author = f"{u['author_label']}: " if u.get("author_label") else ""
        lines.append(f"[unit:{u['id']}] {ts}{author}{u['normalized_text']}")
    return "\n\n".join(lines)


async def extract_chunk(units: list[dict], llm) -> dict:
    """Extract candidates from one chunk of source units via LLM."""
    user = (
        "Extract all decision-relevant elements from the following discussion segment.\n\n"
        + _format_units(units)
    )
    return await llm.extract(_SYSTEM, user, _EXTRACT_TOOL, timeout=90)
