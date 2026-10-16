"""Per-chunk LLM extraction of decision brief candidates."""
from __future__ import annotations

import re

_SYSTEM = """\
You are a precise extraction assistant for decision analysis. Read the discussion segment below and extract structured elements relevant to the decision being made.

Hard rules:
- Only extract what is explicitly stated. Do not infer, generalize, or fabricate.
- For every extracted item, provide the exact verbatim quote from the text (the exact words, not a paraphrase). The quote must come from one of the units you cite.
- A claim is what a participant asserted. It is NOT independently verified fact.
- Silence or lack of objection NEVER implies agreement or a position.
- Units marked [context] are earlier messages shown only so you can read replies correctly. Do NOT extract items from them and do NOT cite them in source_unit_ids.
- Use the unit IDs provided in [unit:...] markers for source_unit_ids.
- Leave lists empty ([]) if nothing of that type appears in this segment.

Positions (stance values):
- "proposed": the person puts an option forward as their own choice ("I'd say X", "or X if we want to do it properly", "I'm not paying more than $400 each"). Listing choices for others to vote on ("Dec 5 or Dec 12. poll") is a question, not a proposal: record no position for it.
- "support": a clear statement of preference for a specific option ("I'll go with X", "X works for me", "I prefer X", "count me in for X", "I moved my thing, X works"); OR a reply that simply names an option in answer to a poll or question ("12th", "Casa Verde", "option A"); OR an explicit "agree" / "yes" / "+1" aimed at a specific proposal in the message just before it; OR a bare "ok" / "yes" / "yep" sent in reply to ONE concrete proposal with nothing else discussed in between (other short replies to that same proposal do not count as in between).
- "conditional": support with a stated condition. Put the condition in `conditions`.
- "oppose": a clear statement against an option.
- "undecided": the person says they have not decided.
- "acknowledged": an indifferent or dismissive reply ("sure whatever", "I'm fine with anything", "I don't care"), a reaction or emoji-only reply ("👍"), a reply that qualifies itself away ("I'm focused on budget but fine"), a generic reply with no single concrete referent ("ok", "yes", "confirmed" after a message that bundles several things or makes claims about others), or a reply that overlaps another speaker's sentence. NEVER count acknowledged as support.
- A reply to a bundled message can still accept the task it assigns to the speaker (see actions); it does not make the speaker support an option.
- Engaging with an option without committing to it ("fine" after the plan is restated, "I want the $180k cap noted as the budget for A") is "acknowledged" on that option.
- "I'll back whichever you pick as long as X" is a constraint from that person (X), not a position on any option.
- participant_label must be the SPEAKER of the cited message. What one person says about another person's view ("Sam is fine with anything") is a CLAIM, not that person's position.
- If a participant revises their stance within this segment, extract ONLY their final stance on that decision. Do not re-extract a stance they have moved away from.
- A later weaker response (an acknowledgment) does not replace an earlier explicit stance.

Options and decisions:
- Every option gets a short `decision` label naming what is being decided ("date", "venue", "transport"). If the KNOWN DECISIONS list below applies, reuse its label exactly. Options for different questions must never share a label.
- If the KNOWN OPTIONS list applies, reuse the exact title for the same option.
- If options are presented with labels ("A) ...", "Option 2"), give the label in `label` and put the substance in `title` ("Nov 16 with Acme"). A position that names only the label ("go with A") uses that option's title.
- `status`: "active" (default, including when an option is merely mentioned again), "dropped" (the group or its owner explicitly took it off the table, it was rejected outright ("lol no") with nobody defending it in this segment, or a fact reported here rules it out (too small, has stairs); the quote must be the message that dropped it), "fallback" (explicitly kept only as a backup), or "reopened" (an option that was dropped earlier is explicitly brought back; the quote must be the message that reopens it).
- For a dropped or fallback option give `status_reason`: the stated reason or condition in a few words ("private room holds only 14", "pending Hal's approval"). Never put private health or money details in it.
- A limit the group sets for itself (a per-person budget cap, a guest cap) is an option of its own decision ("budget", "guest count"), so agreement with it is recorded as positions. Also list it as a constraint.
- A general priority or principle ("compliance first, speed second") is a criterion, not an option.

Constraints, blockers and logistics:
- Venue, room capacity, cost per head, guest count, step-free access, and logistics that directly constrain which option is viable are CONSTRAINTS or CRITERIA, not separate_issues.
- A statement against an option on cost, capacity or approval grounds ("I can't approve B at $215k") is also an objection to that option, with its author.
- Only use separate_issues for topics genuinely unrelated to the decision (for example, a side discussion about something else).
- A blocker or objection that is explicitly resolved in this segment (for example "booked", "sorted", "confirmed", "I moved my thing") must have resolution_status "resolved".
- A blocker is also resolved when the group settles how to handle it and someone commits to the remaining work ("I'll trim the list. 20 or nothing"); the remaining work is an action.
- Use separate_issues for side topics about the same event or group that are not part of this decision: a deferred idea ("photographer — later"), keeping the plan secret, or a disagreement that does not change which option is viable (whether the anniversary is the 8th or the 9th). Do not record such a disagreement as a blocker or claim.
- Ignore off-topic chatter (sports, unrelated errands, jokes) entirely; it is not a separate issue.

Actions:
- Extract an action only when someone commits to it or is asked to do it. A deadline mentioned by a third party ("the venue needs a deposit by Oct 16") is a claim or constraint, not an action.
- First-person commitments ("I'll send the deposit Monday") have owner_label = the speaker and owner_kind "named".
- A task given to someone else ("Leila trims the list by Sunday") has owner_label = that person and owner_kind "named".
- "we'll ..." or "someone should ..." has owner_kind "team" or "unassigned" and NO owner_label.
- A planned checkpoint, review or meeting that someone has to run ("go/no-go checkpoint Oct 30") is an action. Proposing it does not make the proposer its owner: unless someone is named to run it, owner_kind is "unassigned".
- Copy any stated deadline into `due` exactly as written.
- One action per task. "Omar pays for cake and drinks" is two actions: cake and drinks.
- `status`: "open" (default); "done" when the owner reports the task finished ("called them", "I asked, it's fine"); "declined" when the named owner refuses it ("I said cake. not drinks"). For done or declined, repeat the action with its text copied EXACTLY (from the OPEN ITEMS list if it is there) and cite the message that finishes or refuses it.
- `acceptance`: "committed" when the owner says they will do it or accepts it ("I'll trim the list", "yes" to the assignment, "ok" to "is that a yes?", "confirmed" to a summary that names their task); "requested" when someone else assigns it and the owner has not accepted in this segment ("Leila book it"). If the owner accepts later, output the action again, text copied exactly, with acceptance "committed".

Sensitive details:
- If an item involves health, money problems, or other private details, set sensitive=true and write `text` as the requirement only ("step-free access required"), never the private reason. The quote may contain the detail; the text must not.
- A private arrangement between participants (one person covering another's share, a loan) or a personal circumstance is extracted only as ONE claim with sensitive=true, neutral text ("private arrangement between two participants"), citing every message involved. Never as an action, constraint or blocker.
"""

_ITEM_SENSITIVE = {"type": "boolean", "description": "True if the item touches private health, financial or personal details."}

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
                    "required": ["temp_id", "title", "decision", "status", "source_unit_ids", "quote"],
                    "properties": {
                        "temp_id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "decision": {"type": "string", "description": "Short label for the question this option answers, e.g. 'date', 'venue'."},
                        "status": {"type": "string", "enum": ["active", "dropped", "fallback", "reopened"]},
                        "status_reason": {"type": "string", "description": "For dropped or fallback options: the stated reason or condition, a few words."},
                        "label": {"type": "string", "description": "The label the chat uses for this option, e.g. 'A', 'B', '2'."},
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
                        "author_label": {"type": "string"},
                        "sensitive": _ITEM_SENSITIVE,
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
                        "sensitive": _ITEM_SENSITIVE,
                        "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"}
                    }
                }
            },
            "positions": {
                "type": "array",
                "description": "Explicit participant stances on specific options, by the speaker of the cited message. Include only the participant's current/final stance from this segment.",
                "items": {
                    "type": "object",
                    "required": ["participant_label", "option_title", "stance", "source_unit_ids", "quote"],
                    "properties": {
                        "participant_label": {"type": "string"},
                        "option_title": {"type": "string"},
                        "stance": {"type": "string",
                                   "enum": ["proposed", "support", "conditional", "oppose", "undecided", "acknowledged"]},
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
                    "required": ["text", "resolution_status", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "option_titles": {"type": "array", "items": {"type": "string"}},
                        "author_label": {"type": "string"},
                        "sensitive": _ITEM_SENSITIVE,
                        "resolution_status": {
                            "type": "string",
                            "enum": ["open", "addressed", "resolved"],
                            "description": "Set to 'resolved' if explicitly resolved in this segment."
                        },
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
                    "required": ["text", "kind", "resolution_status", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string", "enum": ["open_question", "unresolved_dependency", "missing_info"]},
                        "sensitive": _ITEM_SENSITIVE,
                        "resolution_status": {
                            "type": "string",
                            "enum": ["open", "resolved"],
                            "description": "Set to 'resolved' if the blocker is explicitly cleared in this segment."
                        },
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
                "description": "Tasks someone committed to or was asked to do.",
                "items": {
                    "type": "object",
                    "required": ["text", "owner_kind", "source_unit_ids", "quote"],
                    "properties": {
                        "text": {"type": "string"},
                        "owner_label": {"type": "string"},
                        "owner_kind": {"type": "string", "enum": ["named", "team", "unassigned"]},
                        "due": {"type": "string"},
                        "status": {"type": "string", "enum": ["open", "done", "declined"]},
                        "acceptance": {"type": "string", "enum": ["committed", "requested"]},
                        "sensitive": _ITEM_SENSITIVE,
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

_QUOTED_SECTIONS = (
    "candidate_questions", "options", "criteria", "constraints", "claims",
    "positions", "objections", "blockers", "actions",
)
SECTIONS = _QUOTED_SECTIONS + ("separate_issues",)

_MAX_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Text normalisation shared with the synthesizer's verifier
# ---------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def normalize_quote_text(text: str) -> str:
    """Lower-case, straighten quotes/dashes and collapse whitespace for quote matching."""
    return re.sub(r"\s+", " ", (text or "").translate(_QUOTE_MAP)).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_quote_text(quote)
    return bool(q) and q in normalize_quote_text(text)


def _same_person(a: str, b: str) -> bool:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


# ---------------------------------------------------------------------------
# Prompt construction and result validation
# ---------------------------------------------------------------------------

def _format_units(units: list[dict]) -> str:
    lines = []
    for u in units:
        ts = f"[{u['locator'].get('timestamp_str', '')}] " if u["locator"].get("timestamp_str") else ""
        author = f"{u['author_label']}: " if u.get("author_label") else ""
        ctx = "[context] " if u.get("context_only") else ""
        lines.append(f"[unit:{u['id']}] {ctx}{ts}{author}{u['normalized_text']}")
    return "\n\n".join(lines)


def _known_block(
    known_options: list[dict] | None,
    known_decisions: list[str] | None,
    known_open_items: list[dict] | None = None,
) -> str:
    parts = []
    if known_decisions:
        parts.append("KNOWN DECISIONS (reuse these labels): " + "; ".join(known_decisions))
    if known_options:
        opts = "; ".join(f"{o['title']} [{o.get('decision') or '?'}]" for o in known_options)
        parts.append("KNOWN OPTIONS (reuse exact titles): " + opts)
    if known_open_items:
        items = "; ".join(f"({i['kind']}) {i['text']}" for i in known_open_items)
        parts.append(
            "OPEN ITEMS FROM EARLIER SEGMENTS: " + items + "\n"
            "If this segment shows a blocker or objection resolved, output it again with its text copied EXACTLY and "
            "resolution_status 'resolved'. If it shows an action finished or refused, output the action again with "
            "its text copied EXACTLY and status 'done' or 'declined'. Do not re-list an open item unless this "
            "segment adds something new."
        )
    return ("\n".join(parts) + "\n\n") if parts else ""


def validate_extraction(result: dict, units: list[dict], chunk_index: int = 0) -> dict:
    """Drop items the text cannot support, tag the rest, and record what was dropped.

    * the cited unit ids must be real, non-context units from this chunk
    * the verbatim quote must appear in a cited unit
    * a position's participant must be the speaker of a cited unit (a claim about
      someone else's view is not that person's position)
    * every kept item is tagged with ``_chunk_index`` and ``_seq`` (the earliest
      cited unit's sequence) so the merger can order things chronologically
    """
    by_id = {u["id"]: u for u in units if not u.get("context_only")}
    dropped: list[dict] = []

    for section in SECTIONS:
        kept = []
        for item in result.get(section) or []:
            ids = [i for i in item.get("source_unit_ids", []) if i in by_id]
            if not ids:
                dropped.append({"section": section, "reason": "no_valid_source", "text": item.get("text") or item.get("title", "")})
                continue
            cited = [by_id[i] for i in ids]
            if section in _QUOTED_SECTIONS:
                if not any(quote_in_text(item.get("quote", ""), u["normalized_text"]) for u in cited):
                    dropped.append({"section": section, "reason": "quote_not_in_source", "text": item.get("text") or item.get("title", "")})
                    continue
            if section == "positions":
                authors = [u.get("author_label") for u in cited if u.get("author_label")]
                if authors:
                    match = next((a for a in authors if _same_person(item.get("participant_label", ""), a)), None)
                    if match is None:
                        dropped.append({"section": section, "reason": "speaker_mismatch", "text": item.get("quote", "")})
                        continue
                    item["participant_label"] = match  # canonical label = the unit's author
            if section == "actions" and not item.get("owner_kind"):
                item["owner_kind"] = "named" if item.get("owner_label") else "unassigned"
            if section == "actions" and item.get("owner_kind") != "named":
                item.pop("owner_label", None)
            if section == "actions":
                item.setdefault("status", "open")
                if item["status"] == "open":
                    item.setdefault("acceptance", "committed")
            if section == "options":
                item.setdefault("status", "active")
                item.setdefault("decision", "")
            item["source_unit_ids"] = ids
            item["_chunk_index"] = chunk_index
            item["_seq"] = min(u.get("sequence", 0) for u in cited)
            kept.append(item)
        result[section] = kept

    result["_dropped"] = dropped
    result["_chunk_index"] = chunk_index
    return result


async def extract_chunk(
    units: list[dict],
    llm,
    chunk_index: int = 0,
    known_options: list[dict] | None = None,
    known_decisions: list[str] | None = None,
    known_open_items: list[dict] | None = None,
) -> dict:
    """Extract candidates from one chunk of source units via LLM.

    ``known_options`` / ``known_decisions`` carry titles and labels found in
    earlier chunks so the model reuses them instead of inventing near-duplicates.
    ``known_open_items`` lists blockers/objections still open after earlier chunks
    so a resolution in this chunk is reported against the same wording.
    """
    user = (
        "Extract all decision-relevant elements from the following discussion segment.\n\n"
        + _known_block(known_options, known_decisions, known_open_items)
        + _format_units(units)
    )
    primary_count = sum(1 for u in units if not u.get("context_only"))
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = await llm.extract(_SYSTEM, user, _EXTRACT_TOOL, timeout=90)
            validated = validate_extraction(result, units, chunk_index)
            # If the chunk has substantial content but nothing was extracted, retry once.
            # Empty arrays with no dropped items means the LLM silently skipped the chunk.
            total_items = sum(
                len(validated.get(s, [])) for s in SECTIONS
            ) + len(validated.get("_dropped", []))
            if total_items == 0 and primary_count >= 5 and attempt < _MAX_ATTEMPTS - 1:
                continue
            return validated
        except Exception as exc:  # noqa: BLE001 - retry any transport/schema failure once
            last_exc = exc
    raise RuntimeError(f"extraction failed for chunk {chunk_index}: {last_exc}") from last_exc
