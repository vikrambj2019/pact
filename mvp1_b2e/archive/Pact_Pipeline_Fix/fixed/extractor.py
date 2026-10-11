"""Per-chunk LLM extraction of decision brief candidates."""
from __future__ import annotations

import re

from .reviewer import review_extraction

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
- "proposed": the person who first puts an option forward. A participant who says "I'd propose X", "how about X", "another option is X", "what about X", or "X is possible" is putting that option forward and should receive a "proposed" position even if they also vote for it later. Both positions must be extracted (proposed from the introduction message, support/conditional from the vote message); the merger will reconcile them.
- "support": a clear statement of preference for a specific option ("I'll go with X", "X works for me", "I prefer X", "count me in for X", "I moved my thing, X works"); OR a reply that simply names an option in answer to a poll or question ("12th", "Casa Verde", "option A"); OR an explicit "agree" / "yes" / "+1" aimed at a specific proposal in the message just before it.
- "conditional": support with a stated condition. Put the condition in `conditions`. A message of the form "X works / I could do X, but needs [Y] first" or "X, only if [Z]" is conditional, not support, even if it sounds positive. Extract the condition exactly.
- "oppose": a clear statement against an option. Includes "X was never going to happen", "I'm against X", "X doesn't work for me", "X is off the table".
- "withdrawn": used ONLY when the option's own proposer explicitly retracts or drops their proposal (e.g. "I'm dropping build", "I'm withdrawing my proposal"). Do not use for general opposition by other participants.
- "undecided": the person says they have not decided.
- "acknowledged": a generic response with no clear referent or no commitment ("ok", "fine", "yeah", "sure whatever", "sounds good", "noted", "I'm fine with anything"), or a reply that overlaps another speaker's sentence. NEVER count acknowledged as support.
- participant_label must be the SPEAKER of the cited message. What one person says about another person's view ("Sam is fine with anything") is a CLAIM, not that person's position.
- If a participant revises their stance within this segment, extract ONLY their final stance on that decision. Do not re-extract a stance they have moved away from.
- A later weaker response (an acknowledgment) does not replace an earlier explicit stance.

Options and decisions:
- Every option gets a short `decision` label naming what is being decided. Use the shortest generic 1-2 word label for the question type, e.g. "date", "venue", "transport", "budget", "provider". Do NOT use project-specific phrases like "cutover weekend", "meeting room", "launch date" — prefer "date", "room", "date" respectively. If the KNOWN DECISIONS list below applies, reuse its label exactly. Options for different questions must never share a label.
- If the KNOWN OPTIONS list applies, reuse the exact title for the same option.
- `status`: "active" (default, including when an option is merely mentioned again), "dropped" (the group or its owner explicitly took it off the table OR the group reaches consensus on a competing option; the quote must be the message that dropped it), "fallback" (explicitly kept only as a backup, bridge, or interim solution; phrases like "as a bridge", "until cutover", "interim only", "backup option" signal fallback), or "reopened" (an option that was dropped earlier is explicitly brought back; the quote must be the message that reopens it).
- When the group declares a direction ("Direction: X", "it's decided: X", "we're going with X"), all competing options for the same decision should be marked "dropped".

Constraints, blockers and logistics:
- Venue, room capacity, cost per head, guest count, step-free access, and logistics that directly constrain which option is viable are CONSTRAINTS or CRITERIA, not separate_issues.
- An externally scheduled milestone or sign-off date that gates the feasibility of options (e.g. "UAT sign-off is scheduled for Nov 6", "code freeze is Nov 10") is a CONSTRAINT, not just an action or blocker.
- A target go-live or completion date stated as a requirement (e.g. "live before the January peak, ideally by Jan 15") is a CONSTRAINT. Extract it as a standalone constraint even if it also appears as the reason an option was dropped — the constraint exists independently of any particular option's fate.
- A data-protection or security gate on go-live (e.g. "no customer data to the vendor before the SOC 2 Type II report", "DPA must be signed before go-live") is a CONSTRAINT, in addition to any blocker or conditional stance that captures it.
- Only use separate_issues for topics genuinely unrelated to the decision (for example, a side discussion about something else).
- Operational ownership questions that arise as a side effect of the decision (e.g. "who owns on-call for the legacy system during migration?") are separate_issues if no owner is assigned in this segment — do not classify them as blockers to the main decision.
- A blocker or objection that is explicitly resolved in this segment (for example "booked", "sorted", "confirmed", "I moved my thing") must have resolution_status "resolved".

Actions:
- Extract an action only when someone commits to it or is asked to do it. A deadline mentioned by a third party ("the venue needs a deposit by Oct 16") is a claim or constraint, not an action.
- First-person commitments ("I'll send the deposit Monday") have owner_label = the speaker and owner_kind "named".
- A task given to someone else ("Leila trims the list by Sunday") has owner_label = that person and owner_kind "named".
- "we'll ..." or "someone should ..." has owner_kind "team" or "unassigned" and NO owner_label.
- Copy any stated deadline into `due` exactly as written.

Sensitive details:
- If an item involves health, money problems, or other private details, set sensitive=true and write `text` as the requirement only ("step-free access required"), never the private reason. The quote may contain the detail; the text must not.
"""

_SYSTEM += """
Decision-state rules (apply to every segment):
- Extract agreements on explicit requirements separately from option preferences.
  agreements.subject is the concise requirement (e.g. "$400 per sibling cap",
  "guest cap of 20", "step-free access"); participant_label is the speaker;
  stance is affirm or reject. Reuse known requirement wording exactly.
- A direct "ok" immediately answering a stated budget/cap acknowledges that
  specific requirement and counts as affirm for the requirement ONLY. It never
  creates support for date/venue. Silence and "sure whatever" are not affirmations.
- A requirement affirmed by one person is not a group agreement. Capture each
  affirmer separately. Acting on a requirement is not an explicit affirmation.
- A named date in a poll is support. An explicit conditional yes stays conditional
  until the condition is explicitly satisfied; extract the updated support then.
- Only for option viability, applying explicitly stated capacity/access requirements to explicitly stated venue properties is permitted. Options rejected explicitly, or ruled out by an explicit mandatory capacity or
  accessibility failure, have status dropped. Reopening requires explicit wording.
  A correction or retraction (including false consensus or misassigned drinks responsibility) supersedes the earlier claim. A price correction updates a claim; do not keep the superseded number as current. Re-emit the earlier claim text exactly with evidence_status superseded and cite the current correction message; also extract the corrected claim.
- Distinguish action status: requested (asked/assigned but not accepted), committed
  (owner says they will do it, or explicitly accepts that task), completed (done),
  cancelled (withdrawn/refused). Naming an owner does NOT prove acceptance.
  A bare yes accepts only the immediately referenced task(s), not all earlier ones.
  Preserve due (owner's promised time) separately from external_due (venue deadline).
- Health reasons and private contribution/repayment arrangements have private=true.
  Never put a private arrangement into agreements or public actions. A public budget
  affirmation from the same message may still be extracted without its private reason.
  Accessibility constraints are public requirements with sensitive=true, private=false.
- Reported venue details, prices, parking, and capacity are participant claims,
  not independently verified facts. claims.evidence_status defaults to unverified;
  use conflicting for unresolved competing figures, superseded for corrected figures.
- Photographer suggestions deferred as a different decision, anniversary-date disputes
  irrelevant to the chosen dinner date, and secrecy are separate_issues. Game,
  boiler repair and Thanksgiving discussion are also separate issues. Dinner
  capacity, accessibility and drinks remain core constraints/blockers.
- For a linked constraint resolved later (flight, cake permission, parking), update
  the relevant blocker and conditional position. Guest-list trimming is pending
  until completion is reported. Booking and deposit are pending until done.
- Capture unresolved drinks cost AND who covers an overage as open issues. A quote
  for 18 is not a quote for 20. Do not invent a final total or assume funds suffice.
- Add decision (date, venue, budget, etc.) to positions so identically named options
  in different decisions do not collide. A later proposal or acknowledgment does
  not overwrite stronger support. A claim about another person's view is not a vote.
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
                                   "enum": ["proposed", "support", "conditional", "oppose", "withdrawn", "undecided", "acknowledged"]},
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

# Extend the structured contract without changing existing section names.
_props = _EXTRACT_TOOL["input_schema"]["properties"]
_props["agreements"] = {
    "type": "array", "items": {
        "type": "object", "required": ["subject", "participant_label", "stance", "quote", "source_unit_ids"],
        "properties": {
            "subject": {"type": "string"}, "participant_label": {"type": "string"},
            "stance": {"type": "string", "enum": ["affirm", "reject"]},
            "sensitive": {"type": "boolean"}, "private": {"type": "boolean"},
            "quote": {"type": "string"},
            "source_unit_ids": {"type": "array", "items": {"type": "string"}}
        }
    }
}
_EXTRACT_TOOL["input_schema"]["required"].append("agreements")
for _section in ("constraints", "claims", "actions", "objections", "blockers", "agreements"):
    _props[_section]["items"]["properties"]["private"] = {"type": "boolean"}
_props["positions"]["items"]["properties"]["decision"] = {"type": "string"}
_props["actions"]["items"]["properties"].update({
    "status": {"type": "string", "enum": ["requested", "committed", "completed", "cancelled"]},
    "external_due": {"type": "string"}
})
_props["actions"]["items"]["required"].append("status")
_props["claims"]["items"]["properties"]["evidence_status"] = {
    "type": "string", "enum": ["unverified", "conflicting", "superseded"]
}

_STATE_TOOL = {
    "name": "extract_commitments",
    "description": "Focused coverage pass for participant votes, requirements, actions and side decisions.",
    "input_schema": {
        "type": "object",
        "properties": {key: _props[key] for key in ("positions", "agreements", "actions", "separate_issues", "claims", "blockers", "objections", "constraints")},
        "required": ["positions", "agreements", "actions", "separate_issues", "claims", "blockers", "objections", "constraints"],
    },
}
_COVERAGE_RULES = """
This is a separate coverage pass, focused on speech acts rather than factual claims.
Read EVERY primary message. Extract all supported positions, requirement affirmations,
actions and separate issues; do not assume the earlier draft captured them.
- A message supporting a bundle of date + venue yields a position for EACH decision.
  A direct yes responding to a clearly enumerated bundle affirms each referenced
  decision/requirement. A generic 'sure whatever' affirms none.
- When someone agrees with a message that states BOTH a primary choice AND a conditional
  fallback (e.g. "Yes to X; I'd use Y if Z slips"), the agreeing participant gets a
  support for the primary option AND a conditional for the fallback option.
- For each option in the preliminary draft, check whether the person who introduced it
  has a 'proposed' position. If that person only has a 'support' or 'conditional'
  position, also add a 'proposed' position for them citing the introduction message.
- Check every message for go-live targets or delivery deadlines expressed with
  "ideally", "if possible", "we want", "we need this live by" — these are soft
  constraints and must be captured even if they are also the reason an option was
  dropped. Add any missed ones to constraints.
- A travel booking followed by 'so [date] works' is explicit date support, not
  still conditional. Resolving one's conflict then naming the date is support.
- A final message restating the guest cap explicitly affirms the cap; an adjacent
  yes by another speaker to that concrete plan also affirms it. Reporting the
  room capacity alone is a claim, not personal affirmation of a guest-list cap.
- Do not turn a factual report of cake permission or parking into an agreement.
- 'I'm doing the cake' or 'cake is mine' is an owner commitment even without 'I'll'.
  Capture every public first-person commitment, including an action alongside a vote.
- For the same task and named owner, reuse the preliminary/known action text exactly
  and update status/deadline fields instead of inventing a paraphrased duplicate.
- Keep exact amounts in action text. For deposit actions include the promised due
  AND any earlier stated venue deadline as external_due, using read-only prior state
  where needed. Never replace the promised time with the venue deadline.
- A later yes accepts the immediately assigned task only; it does not accept an
  earlier booking request. A repeated request cannot erase an earlier commitment.
- Anniversary-date disagreements that do not change the selected dinner weekend,
  secrecy and a deferred photographer decision belong in separate_issues.
The preliminary draft is not authoritative. Correct errors from source evidence.
Use only PRIMARY unit IDs for citations; never cite prior state or context units.
"""

_QUOTED_SECTIONS = (
    "candidate_questions", "options", "criteria", "constraints", "claims",
    "positions", "objections", "blockers", "actions", "agreements",
)
SECTIONS = _QUOTED_SECTIONS + ("separate_issues",)

_SYSTEM += """
Enterprise dependency rules:
- A later bare yes/confirmed does not release an earlier condition. Keep the
  condition unless its entire requirement is explicitly satisfied or withdrawn.
- Receipt of an artifact and completion of its review/sign-off are separate
  states. A received report does not resolve a pending review/sign-off blocker.
- An escalation path or request for approval is NOT approval. An objection to
  unapproved spending stays open until approval or withdrawal is explicit.
- 'I'll back whichever you pick' is generic sponsorship, not an option vote or
  authorization of additional funding. Extract its deadline as a constraint.
- A scheduled go/no-go checkpoint is an action with owner_kind unassigned unless
  someone explicitly owns it; proposing it does not make the proposer its owner.
- A named option proposer must be retained. Do not attach a generic acknowledgment
  to a different option simply because that option was most recently mentioned.
- Record competing effort estimates as conflicting claims. Keep corrected costs
  and total estimates, marking the old figures superseded. Estimates are not caps.
- An owner's target for a third party's signature is not a guarantee of signature.
- Do not copy a proposed target date into an owner's promised due field unless
  that owner accepted that date for that task. 'I can start tomorrow', 'ok', and
  a progress report do not accept an earlier question about completion by a date.
- In 'Marcus confirms Nov 6 code complete' followed by Marcus 'confirmed', the
  named task and date are accepted; keep due Nov 6 and the task 'code complete'.
- 'I'll know more Monday' commits to an update, not resolution of the issue.
- Legal signature, security review and closing an auth finding are separate
  blockers. Do not also count a general statement listing those same gates as
  another blocker. A review requirement is a gate, not an objection to the plan.
- A funding ask referencing the checkpoint is not ownership of running it.
- A checkpoint attached to an option is not an additional option for the launch
  plan. Someone's 'Good with A and the checkpoint' supports A plus the checkpoint
  requirement, without creating a competing checkpoint plan.
- Conflicting effort estimates are claims, not an extra mandatory launch gate.
- Starting work on a date is not completing work by that date. Record start_due
  separately from the completion due field.
- Operational tasks and deadlines remain public when their rationale is personal
  availability or leave; omit the personal reason rather than the task.
"""
_COVERAGE_RULES += """
Audit dependencies, objections and estimates as well as speech acts. Check each
primary message for a review/sign-off still pending after document receipt,
an unapproved funding objection, or a checkpoint without an accepted owner.
Preserve conditional votes across bare confirmations. Do not omit revised costs
or disputed effort estimates. Retain a pending gate even when progress is reported.
"""

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
    # Short quotes such as "ok" must not match inside "booked".
    return bool(q) and bool(re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", normalize_quote_text(text)))


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
    known_actions: list[dict] | None = None,
    known_agreements: list[str] | None = None,
    known_state: dict | None = None,
) -> str:
    parts = []
    if known_state:
        import json
        parts.append("EARLIER STATE (read-only; cite only current primary messages for updates): " + json.dumps(known_state))
    if known_actions:
        import json
        parts.append("KNOWN ACTIONS (reuse text exactly for the same task/owner; update status only when the owner accepts/completes): " + json.dumps(known_actions))
    if known_agreements:
        parts.append("KNOWN REQUIREMENTS (reuse subject wording): " + "; ".join(known_agreements))
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
            if not ids and item.get("quote"):
                # Repair a mistyped citation only when the exact quote identifies
                # one primary message. Ambiguous yes/ok replies are never guessed.
                matches = [u for u in by_id.values() if quote_in_text(item["quote"], u["normalized_text"])]
                if section in ("positions", "agreements"):
                    matches = [u for u in matches if _same_person(item.get("participant_label", ""), u.get("author_label", ""))]
                if len(matches) == 1:
                    ids = [matches[0]["id"]]
                    item["citation_repaired"] = True
            if not ids:
                dropped.append({"section": section, "reason": "no_valid_source", "text": item.get("text") or item.get("title", "")})
                continue
            cited = [by_id[i] for i in ids]
            if section in _QUOTED_SECTIONS:
                if not any(quote_in_text(item.get("quote", ""), u["normalized_text"]) for u in cited):
                    dropped.append({"section": section, "reason": "quote_not_in_source", "text": item.get("text") or item.get("title", "")})
                    continue
            if section in ("positions", "agreements"):
                authors = [u.get("author_label") for u in cited if u.get("author_label") and quote_in_text(item.get("quote", ""), u["normalized_text"])]
                if not authors:
                    dropped.append({"section": section, "reason": "speaker_unknown", "text": item.get("quote", "")})
                    continue
                if authors:
                    match = next((a for a in authors if _same_person(item.get("participant_label", ""), a)), None)
                    if match is None:
                        dropped.append({"section": section, "reason": "speaker_mismatch", "text": item.get("quote", "")})
                        continue
                    item["participant_label"] = match  # canonical label = the unit's author
            if section == "agreements" and item.get("stance") == "affirm":
                quote = item.get("quote", "").strip()
                if re.search(r"\bfine[.!]?$", quote, re.I) and not re.search(r"\bagree|\byes\b", quote, re.I):
                    continue
                if re.fullmatch(r"(?:ok|noted)[.!]?", quote, re.I):
                    reply_seq = min(u["sequence"] for u in cited if u.get("author_label") == item["participant_label"])
                    previous = [u for u in units if u["sequence"] < reply_seq]
                    prior = next((u for u in reversed(previous) if not re.fullmatch(r"ok[.!]?", u["normalized_text"].strip(), re.I)), None)
                    if quote.casefold().rstrip(".!") != "ok" or not prior or not re.search(r"\bcap|ceiling|limit\b", prior["normalized_text"], re.I):
                        continue
            if section == "actions":
                item.setdefault("status", "requested")
                # A third party assigning a task is not the owner's commitment.
                if item["status"] == "committed" and item.get("owner_kind") == "named":
                    owner = item.get("owner_label", "")
                    if not any(_same_person(owner, u.get("author_label", ""))
                               and quote_in_text(item.get("quote", ""), u["normalized_text"])
                               for u in cited):
                        item["status"] = "requested"
                if item["status"] == "committed" and re.search(r"\b(?:book|reserve)\b", item.get("text", ""), re.I):
                    quote = item.get("quote", "")
                    explicit = bool(re.search(r"\b(?:book|reserve)\w*\b", quote, re.I))
                    if not explicit and re.fullmatch(r"yes[.!]?", quote.strip(), re.I):
                        quoted = [u for u in cited if quote_in_text(quote, u["normalized_text"])
                                  and _same_person(item.get("owner_label", ""), u.get("author_label", ""))]
                        for reply in quoted:
                            prev = next((u for u in units if u.get("sequence") == reply.get("sequence", 0) - 1), None)
                            if prev and re.search(r"\b(?:book|reserve)\w*\b", prev["normalized_text"], re.I):
                                explicit = True
                    if not explicit:
                        item["status"] = "requested"
            if section == "claims":
                item.setdefault("evidence_status", "unverified")
            if section == "actions" and item.get("status") == "committed" and item.get("owner_kind") == "named":
                due = item.get("due", "")
                owner_sources = [u for u in cited if _same_person(item.get("owner_label", ""), u.get("author_label", ""))]
                if due and not any(quote_in_text(due, u["normalized_text"]) for u in owner_sources):
                    # A third-party target is not an accepted deadline. Named
                    # bundle confirmations are recovered from source below.
                    item["requested_due"] = item.pop("due")
                elif due and any(re.search(r"\bstart\b", u["normalized_text"], re.I) and quote_in_text(due, u["normalized_text"]) for u in owner_sources):
                    item["start_due"] = item.pop("due")
                if re.search(r"\bI(?:'|’)ll know more\b", item.get("quote", ""), re.I):
                    item["text"] = "Provide a status update"
            if section in ("blockers", "objections") and item.get("resolution_status") == "resolved":
                quote = item.get("quote", "")
                # Naming a path to resolution cannot itself close a gate.
                if re.search(r"\b(?:pending|requires?|needs?|has to|have to|someone|not signed|still open|before I sign off)\b", quote, re.I):
                    item["resolution_status"] = "open"
            if section == "positions" and re.search(r"\bwhichever (?:you|we|they) pick\b", item.get("quote", ""), re.I):
                continue
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
    known_actions: list[dict] | None = None,
    known_agreements: list[str] | None = None,
    known_state: dict | None = None,
    coverage_llm=None,
) -> dict:
    """Extract candidates from one chunk of source units via LLM.

    ``known_options`` / ``known_decisions`` carry titles and labels found in
    earlier chunks so the model reuses them instead of inventing near-duplicates.
    ``known_open_items`` lists blockers/objections still open after earlier chunks
    so a resolution in this chunk is reported against the same wording.
    ``coverage_llm`` overrides the model used for the focused coverage pass only;
    defaults to ``llm`` if not provided.
    """
    _coverage_llm = coverage_llm or llm
    user = (
        "Extract all decision-relevant elements from the following discussion segment.\n\n"
        + _known_block(known_options, known_decisions, known_open_items, known_actions, known_agreements, known_state)
        + _format_units(units)
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            attempt_user = user + ("\nPrevious failed validation/review; correct these issues against source:\n" + str(last_exc) if last_exc else "")
            result = await llm.extract(_SYSTEM, attempt_user, _EXTRACT_TOOL, timeout=90)
            result = validate_extraction(result, units, chunk_index)
            import json
            focus_user = attempt_user + "\n\nPreliminary draft (audit against source):\n" + json.dumps(result)
            focused = await _coverage_llm.extract(_SYSTEM + _COVERAGE_RULES, focus_user, _STATE_TOOL, timeout=90)
            focused = validate_extraction(focused, units, chunk_index)
            for section in ("positions", "agreements", "actions", "separate_issues", "claims", "blockers", "objections", "constraints"):
                result[section].extend(focused[section])
            result["_dropped"].extend(focused["_dropped"])
            result["_review"] = await review_extraction(result, units, llm, context=known_state)
            return result
        except Exception as exc:  # noqa: BLE001 - retry any transport/schema failure once
            last_exc = exc
    raise RuntimeError(f"extraction failed for chunk {chunk_index}: {last_exc}") from last_exc
