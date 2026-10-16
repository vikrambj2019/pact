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
- "proposed": the person who first puts an option forward. A sender who opens with a neutral list of options gets a "proposed" position for each option they put forward. Someone who LATER picks an option already on the table is "support", not "proposed".
- "support": a clear statement of preference for a specific option ("I'll go with X", "X works for me", "I prefer X", "I would pick X", "I'd go with X", "X for me", "count me in for X", "I moved my thing, X works"); OR a reply that simply names an option in answer to a poll or question ("12th", "Casa Verde", "option A"); OR an explicit "agree" / "yes" / "+1" aimed at a specific proposal in the message just before it.
- "conditional": support with a stated condition. Put the condition in `conditions`.
- "oppose": a clear statement against an option.
- "undecided": the person says they have not decided.
- "acknowledged": a generic response with no clear referent or no commitment ("ok", "fine", "yeah", "sure whatever", "sounds good", "noted", "I'm fine with anything"), or a reply that overlaps another speaker's sentence. NEVER count acknowledged as support.
- A reason an option does not work ("Feb 9 clashes with the audit", "that is over our budget", "we have a rule against X", "I do not have capacity for that") is also an OBJECTION against that option, in addition to the speaker's position. Use resolution_status "open" unless the same segment resolves it. A request for missing information that nobody answers ("I just want the build cost on the table") is a blocker (kind "open_question").
- participant_label must be the SPEAKER of the cited message. What one person says about another person's view ("Sam is fine with anything") is a CLAIM, not that person's position.
- If a participant revises their stance within this segment, extract ONLY their final stance on that decision. Do not re-extract a stance they have moved away from.
- A later weaker response (an acknowledgment) does not replace an earlier explicit stance.

Options and decisions:
- Every option gets a short `decision` label naming what is being decided ("date", "venue", "transport"). If the KNOWN DECISIONS list below applies, reuse its label exactly. Options for different questions must never share a label.
- If the KNOWN OPTIONS list applies, reuse the exact title for the same option.
- `status`: "active" (default, including when an option is merely mentioned again), "dropped" (the group or its owner explicitly took it off the table; the quote must be the message that dropped it), "fallback" (explicitly kept only as a backup), or "reopened" (an option that was dropped earlier is explicitly brought back; the quote must be the message that reopens it).

Constraints, blockers and logistics:
- Venue, room capacity, cost per head, guest count, step-free access, and logistics that directly constrain which option is viable are CONSTRAINTS or CRITERIA, not separate_issues.
- Only use separate_issues for topics genuinely unrelated to the decision (for example, a side discussion about something else).
- A blocker or objection that is explicitly resolved in this segment (for example "booked", "sorted", "confirmed", "I moved my thing") must have resolution_status "resolved".

Actions:
- Extract an action only when someone commits to it or is asked to do it. A deadline mentioned by a third party ("the venue needs a deposit by Oct 16") is a claim or constraint, not an action.
- First-person commitments ("I'll send the deposit Monday") have owner_label = the speaker and owner_kind "named".
- A task given to someone else ("Leila trims the list by Sunday") has owner_label = that person and owner_kind "named".
- "we'll ..." or "someone should ..." has owner_kind "team" or "unassigned" and NO owner_label.
- A person who is asked and declines, or limits what they accept ("I can't take it the week of the 9th"), owns only the part they accepted. Extract the part nobody took as its OWN action with owner_kind "unassigned" and wording that names it ("Cover the Monday refresh the week of the 9th"). Never leave the declined part with the person who declined it.
- Copy any stated deadline into `due` exactly as written.

Sensitive details:
- If an item involves health, money problems, or other private details, set sensitive=true and write `text` as the requirement only ("step-free access required"), never the private reason. The quote may contain the detail; the text must not.

Source types (the unit header tells you which one you are reading):
- EMAIL units: each unit is ONE email's own new text, with its sender, To and Cc shown. Older emails it quoted have already been split out as their own earlier units. The speaker is the SENDER. People on To are being addressed; people on Cc are only informed.
  * Being on To or Cc is never agreement. Silence is never a position, and never report that someone "did not respond".
  * A request addressed to a named person ("@Sam please confirm...", "Sam, can you...") is an action for that person: owner_label = that person, owner_kind "named". It does not mean they accepted.
  * A forwarding note ("FYI", "see below", "keep it to yourself", "we need to discuss this") is the forwarder's own words. It is not a position on the forwarded content and not the original author's view.
  * An email proposing a decision, approval, funding or plan is an option; the sender is its proposer. A reply that counters or rewrites it is a position (oppose or conditional) on that option, and may add an alternative option.
  * Units marked [RESTRICTED] were sent with a request to keep them private. Extract normally, including the sender's position and any concern they raise as an objection; the system handles who may see the result.
  * A unit whose recipients are unknown is a cover note. Treat its author as unknown and do not attribute positions to anyone.
- MEETING TRANSCRIPT units: the speaker is the person named on the unit. Spoken text is loose: "this one", "that one", "the second" often point at a screen share you cannot see. If you cannot tell what is being referred to, do NOT guess: extract only what the words themselves state. Names of vendors and tools may be mis-transcribed; copy them as written. A unit marked with a gap means minutes of the meeting are missing before it. Do not assume anything about what was said in the gap.
- Spoken backchannel ("yeah", "mhm", "sure", "okay") is acknowledged at most, never support. When it answers a specific proposal ("so that one then?" / "Yeah, yeah."), record it as an "acknowledged" position on that option so the summary can show who only acknowledged. A facilitator's question ("so that one then?") is not a position.
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

def _short(names: list[str], limit: int = 6) -> str:
    names = [n for n in names if n]
    if len(names) <= limit:
        return "; ".join(names) or "none"
    return "; ".join(names[:limit]) + f"; +{len(names) - limit} more"


def _unit_prefix(u: dict) -> str:
    loc = u.get("locator") or {}
    ts = f"[{loc['timestamp_str']}] " if loc.get("timestamp_str") else ""
    author = f"{u['author_label']}: " if u.get("author_label") else ""
    if loc.get("source_type") == "email":
        kind = loc.get("msg_kind", "email")
        if loc.get("audience_unknown"):
            route = "recipients unknown"
        else:
            route = f"To: {_short(loc.get('to', []))}"
            if loc.get("cc"):
                route += f" | Cc: {_short(loc['cc'])}"
        flag = " [RESTRICTED]" if loc.get("restricted") else ""
        subj = f" Subject: {loc['subject']}." if loc.get("subject") else ""
        return f"[email {kind}] {ts}{author.rstrip(': ')} -> {route}{flag}.{subj} "
    gap = ""
    if loc.get("gap_before_s"):
        gap = f"[GAP: about {round(loc['gap_before_s'] / 60)} minutes of the meeting are missing before this] "
    return f"{gap}{ts}{author}"


def _format_units(units: list[dict]) -> str:
    lines = []
    for u in units:
        ctx = "[context] " if u.get("context_only") else ""
        lines.append(f"[unit:{u['id']}] {ctx}{_unit_prefix(u)}{u['normalized_text']}")
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
            "If this segment shows one of them resolved, output it again with its text copied EXACTLY and "
            "resolution_status 'resolved'. Do not re-list an open item unless this segment adds something new."
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
    last_exc: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            result = await llm.extract(_SYSTEM, user, _EXTRACT_TOOL, timeout=150)
            return validate_extraction(result, units, chunk_index)
        except Exception as exc:  # noqa: BLE001 - retry any transport/schema failure once
            last_exc = exc
    raise RuntimeError(f"extraction failed for chunk {chunk_index}: {last_exc}") from last_exc
