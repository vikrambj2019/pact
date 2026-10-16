"""Staged extraction orchestration — runs extract_brief and synthesize_brief jobs."""
from __future__ import annotations

import json
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.models import (
    BriefRevision, Event, Job, JobStatus, JobType, SourceUnit, Workspace,
)
from .audience import attach_audience, cited_unit_ids, filter_for_viewer, scope_line
from .extractor import quote_in_text
from .merger import _norm_text, _same_option, _similar


# ---------------------------------------------------------------------------
# Job 1: extract_brief  (chunk → per-chunk extract → merge → candidate questions)
# ---------------------------------------------------------------------------

async def run_extract_brief(payload: dict, session: AsyncSession) -> dict:
    from .chunker import chunk_units
    from .extractor import extract_chunk
    from .llm import LLMClient
    from .merger import merge_extractions

    workspace_id = uuid.UUID(payload["workspace_id"])

    result = await session.execute(
        select(SourceUnit)
        .where(
            SourceUnit.workspace_id == workspace_id,
            SourceUnit.duplicate_of.is_(None),
        )
        .order_by(SourceUnit.sequence)
    )
    units = result.scalars().all()
    if not units:
        return {"candidate_questions": [], "merged_draft": {}, "unit_count": 0, "chunk_count": 0}

    units_data = [
        {
            "id": str(u.id),
            "sequence": u.sequence,
            "author_label": u.author_label,
            "normalized_text": u.normalized_text,
            "locator": u.locator or {},
        }
        for u in units
    ]
    unit_sequence = {u["id"]: u["sequence"] for u in units_data}

    llm = LLMClient()
    chunks = chunk_units(units_data)

    # Chunks run in order so each one is told which options and decision labels
    # earlier chunks already found; the model reuses them instead of inventing
    # near-duplicates that the merger would then have to reconcile.
    chunk_results = []
    known_options: list[dict] = []
    known_decisions: list[str] = []
    open_items: dict[str, dict] = {}          # normalised text -> {"kind", "text"}
    for i, chunk in enumerate(chunks):
        extracted = await extract_chunk(
            chunk, llm, chunk_index=i,
            known_options=list(known_options) or None,
            known_decisions=list(known_decisions) or None,
            known_open_items=list(open_items.values()) or None,
        )
        chunk_results.append(extracted)
        for opt in extracted.get("options", []):
            if not any(k["title"].lower() == opt["title"].lower() for k in known_options):
                known_options.append({"title": opt["title"], "decision": opt.get("decision", "")})
            label = opt.get("decision")
            if label and label not in known_decisions:
                known_decisions.append(label)
        for section, kind in (("blockers", "blocker"), ("objections", "objection")):
            for item in extracted.get(section, []):
                key = _norm_text(item["text"])
                if item.get("resolution_status", "open") == "open":
                    open_items[key] = {"kind": kind, "text": item["text"]}
                else:
                    open_items.pop(key, None)

    merged = merge_extractions(chunk_results, unit_sequence=unit_sequence)

    return {
        "candidate_questions": merged["candidate_questions"],
        "merged_draft": merged,
        "unit_count": len(units_data),
        "chunk_count": len(chunks),
        "dropped_item_count": len(merged.get("dropped_items", [])),
    }


# ---------------------------------------------------------------------------
# Job 2: synthesize_brief  (confirmed question + merged draft → BriefRevision)
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """\
You are a decision brief synthesizer. Given a set of extraction candidates from a discussion and a confirmed decision question, produce a clean, structured decision brief.

Rules:
- The question may be settled through several sub-decisions (for example date, venue, budget). Keep every item that bears on the question or on any of its sub-decisions. Venue, capacity, cost, access and logistics are constraints or criteria, never separate issues.
- Use separate_issues only for topics genuinely unrelated to the decision.
- Do not drop candidates: every option, position, objection, blocker and action in the candidates must appear in your output, merging only true duplicates. Do not invent items, owners, dates or sources.
- Keep each option's `decision` and `status`, each position's `stance`, each action's `owner_kind`, `owner_label` and `due`, and each objection's and blocker's `resolution_status` exactly as given. Never turn "acknowledged" into support. Never assign an owner the candidates do not name.
- Keep `sensitive` flags. For sensitive items state only the requirement, never the private reason.
- For claims: evidence_status is "supported_within_scope" only when another participant gives evidence or a reason; a bare assertion is "unverified"; use "conflicting" when participants give different values for the same thing.
- Do NOT include an "id" field on any item — IDs are assigned by the server.
- All source_unit_ids you cite must come from the candidates provided — never invent IDs.
- Do not output emerging_preference; the system computes it from positions.
- Keep all text concise and grounded in the source material.
"""

_SENSITIVE = {"type": "boolean"}

_SYNTH_TOOL = {
    "name": "produce_brief",
    "description": "Produce a clean structured decision brief. Do NOT include id fields.",
    "input_schema": {
        "type": "object",
        "required": ["criteria", "constraints", "options", "claims", "positions",
                     "objections", "blockers", "separate_issues", "actions"],
        "properties": {
            "criteria": {"type": "array", "items": {
                "type": "object", "required": ["name", "source_unit_ids"],
                "properties": {
                    "name": {"type": "string"}, "description": {"type": "string"},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "constraints": {"type": "array", "items": {
                "type": "object", "required": ["text", "kind", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["mandatory", "negotiable", "unknown"]},
                    "basis": {"type": "string"},
                    "sensitive": _SENSITIVE,
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "options": {"type": "array", "items": {
                "type": "object", "required": ["title", "decision", "status", "source_unit_ids"],
                "properties": {
                    "title": {"type": "string"}, "description": {"type": "string"},
                    "decision": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "dropped", "fallback"]},
                    "arguments_for": {"type": "array", "items": {"type": "string"}},
                    "arguments_against": {"type": "array", "items": {"type": "string"}},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "claims": {"type": "array", "items": {
                "type": "object",
                "required": ["text", "kind", "evidence_status", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["factual", "predictive", "causal", "priority", "constraint", "other"]},
                    "author_label": {"type": "string"},
                    "sensitive": _SENSITIVE,
                    "evidence_status": {"type": "string",
                                        "enum": ["unverified", "supported_within_scope",
                                                 "contradicted_within_scope", "conflicting", "unresolved"]},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "positions": {"type": "array", "items": {
                "type": "object",
                "required": ["participant_label", "option_title", "stance", "source_unit_ids"],
                "properties": {
                    "participant_label": {"type": "string"},
                    "option_title": {"type": "string"},
                    "stance": {"type": "string",
                               "enum": ["proposed", "support", "conditional", "oppose",
                                        "undecided", "acknowledged", "unknown"]},
                    "proposer": {"type": "boolean"},
                    "conditions": {"type": "string"},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "objections": {"type": "array", "items": {
                "type": "object",
                "required": ["text", "resolution_status", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "option_titles": {"type": "array", "items": {"type": "string"}},
                    "author_label": {"type": "string"},
                    "sensitive": _SENSITIVE,
                    "moot": {"type": "boolean"},
                    "resolution_status": {"type": "string",
                                          "enum": ["open", "addressed", "resolved"]},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "blockers": {"type": "array", "items": {
                "type": "object",
                "required": ["text", "kind", "resolution_status", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["open_question", "unresolved_dependency", "missing_info"]},
                    "sensitive": _SENSITIVE,
                    "resolution_status": {"type": "string", "enum": ["open", "resolved"]},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "separate_issues": {"type": "array", "items": {
                "type": "object", "required": ["title", "rationale", "source_unit_ids"],
                "properties": {
                    "title": {"type": "string"}, "rationale": {"type": "string"},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "actions": {"type": "array", "items": {
                "type": "object", "required": ["text", "owner_kind", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"}, "owner_label": {"type": "string"},
                    "owner_kind": {"type": "string", "enum": ["named", "team", "unassigned"]},
                    "due": {"type": "string"},
                    "sensitive": _SENSITIVE,
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
        }
    }
}

_BRIEF_SECTIONS = ("criteria", "constraints", "options", "claims",
                   "positions", "objections", "blockers", "separate_issues", "actions")


# ---------------------------------------------------------------------------
# Safety net: never let the synthesis step lose or rewrite identity-bearing facts
# ---------------------------------------------------------------------------

def _restore_dropped(brief_body: dict, merged: dict) -> list[str]:
    """Re-add merged-draft items the LLM dropped and restore fields it must not change.

    Options, positions, objections, blockers and actions feed the status counts, so
    their stance / owner / due / resolution / status come from the merged draft, not
    from the synthesis model. Run BEFORE ``_assign_server_ids`` (titles still present).
    Returns human-readable notes about what was restored or corrected.
    """
    notes: list[str] = []

    # options
    opts = brief_body.setdefault("options", [])
    for cand in merged.get("options", []):
        match = next((o for o in opts if _same_option(o.get("title", ""), cand["title"])), None)
        if match is None:
            opts.append({**{k: cand[k] for k in ("title", "description", "decision", "status", "source_unit_ids") if k in cand},
                         "restored": True})
            notes.append(f"restored option '{cand['title']}'")
            continue
        for f in ("decision", "status"):
            if cand.get(f) and match.get(f) != cand[f]:
                notes.append(f"option '{match.get('title')}': {f} {match.get(f)!r} -> {cand[f]!r}")
                match[f] = cand[f]

    # positions
    positions = brief_body.setdefault("positions", [])
    for cand in merged.get("positions", []):
        who = cand["participant_label"].strip().lower()
        match = next((p for p in positions
                      if p.get("participant_label", "").strip().lower() == who
                      and _same_option(p.get("option_title", ""), cand["option_title"])), None)
        if match is None:
            positions.append({**{k: cand[k] for k in ("participant_label", "option_title", "stance", "conditions",
                                                       "proposer", "source_unit_ids") if k in cand},
                              "restored": True})
            notes.append(f"restored position {cand['participant_label']} / {cand['option_title']}")
        elif match.get("stance") != cand["stance"]:
            notes.append(f"position {cand['participant_label']} / {cand['option_title']}: stance "
                         f"{match.get('stance')!r} -> {cand['stance']!r}")
            match["stance"] = cand["stance"]

    # objections and blockers: status comes from chronology, not from the LLM
    for section in ("objections", "blockers"):
        items = brief_body.setdefault(section, [])
        for cand in merged.get(section, []):
            match = next((i for i in items if _similar(i.get("text", ""), cand["text"])), None)
            if match is None:
                keep = ("text", "kind", "option_titles", "author_label", "moot", "sensitive",
                        "resolution_status", "source_unit_ids")
                items.append({**{k: cand[k] for k in keep if k in cand}, "restored": True})
                notes.append(f"restored {section[:-1]} '{cand['text'][:60]}'")
                continue
            if cand.get("resolution_status") and match.get("resolution_status") != cand["resolution_status"]:
                notes.append(f"{section[:-1]} '{match['text'][:50]}': status "
                             f"{match.get('resolution_status')!r} -> {cand['resolution_status']!r}")
                match["resolution_status"] = cand["resolution_status"]
            if cand.get("moot"):
                match["moot"] = True

    # actions: the owner is whatever the chat named, never an LLM guess
    actions = brief_body.setdefault("actions", [])
    for cand in merged.get("actions", []):
        match = next((a for a in actions if _similar(a.get("text", ""), cand["text"])), None)
        if match is None:
            keep = ("text", "owner_label", "owner_kind", "due", "sensitive", "source_unit_ids")
            actions.append({**{k: cand[k] for k in keep if k in cand}, "restored": True})
            notes.append(f"restored action '{cand['text'][:60]}'")
            continue
        kind = cand.get("owner_kind") or ("named" if cand.get("owner_label") else "unassigned")
        label = cand.get("owner_label") if kind == "named" else None
        if match.get("owner_kind") != kind or match.get("owner_label") != label:
            notes.append(f"action '{match['text'][:50]}': owner {match.get('owner_label')!r} -> {label!r}")
        match["owner_kind"] = kind
        if label:
            match["owner_label"] = label
        else:
            match.pop("owner_label", None)
        if cand.get("due"):
            match["due"] = cand["due"]

    # sensitivity flags are sticky in every section
    for section in ("constraints", "claims", "objections", "blockers", "actions"):
        for cand in merged.get(section, []):
            if cand.get("sensitive"):
                for item in brief_body.get(section, []):
                    if _similar(item.get("text", ""), cand["text"]):
                        item["sensitive"] = True
    return notes


# ---------------------------------------------------------------------------
# Exec summary — deterministic parts
# ---------------------------------------------------------------------------

_BANNED_WORDS = [
    "consensus", "agreed", "decided", "leans",
    "converging", "converged", "effectively", "eliminated",
]

_BANNED_PLACEHOLDERS = ["(source)", "(cite)", "(id)", "[source]"]

_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _BANNED_WORDS) + r")\b", re.IGNORECASE)


def _check_banned_words(text: str) -> list[str]:
    """Whole-word match, so 'disagreed', 'undecided' and 'cleans' are not flagged."""
    found = sorted({m.group(1).lower() for m in _BANNED_RE.finditer(text)})
    lower = text.lower()
    found += [p for p in _BANNED_PLACEHOLDERS if p.lower() in lower]
    return found


def _strip_banned_lines(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in text.split("\n"):
        if _check_banned_words(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _named_owner(action: dict) -> bool:
    return bool(action.get("owner_label")) and action.get("owner_kind", "named") == "named"


def _compute_status_line(brief_body: dict) -> str:
    """Compute the status counts in Python so the LLM cannot get them wrong."""
    options = brief_body.get("options", [])
    positions = brief_body.get("positions", [])
    blockers = brief_body.get("blockers", [])
    objections = brief_body.get("objections", [])
    actions = brief_body.get("actions", [])

    opt_by_id = {o["id"]: o for o in options if "id" in o}
    active = [o for o in options if o.get("status") != "dropped"]
    dropped = len(options) - len(active)

    stances: dict[str, dict[str, list[str]]] = {}
    for p in positions:
        o = opt_by_id.get(p.get("option_id", ""))
        if o is None or o.get("status") == "dropped":
            continue
        stances.setdefault(o["id"], {}).setdefault(p["stance"], []).append(p["participant_label"])

    parts: list[str] = [f"{len(active)} option{'s' if len(active) != 1 else ''}"
                        + (f" ({dropped} dropped)" if dropped else "")]
    for o in active:
        st = stances.get(o.get("id", ""), {})
        bits = []
        if st.get("support"):
            bits.append(f"explicit support from {', '.join(_unique(st['support']))} ({len(_unique(st['support']))})")
        if st.get("conditional"):
            bits.append(f"conditional: {', '.join(_unique(st['conditional']))}")
        if st.get("acknowledged"):
            bits.append(f"acknowledged only: {', '.join(_unique(st['acknowledged']))}")
        if st.get("oppose"):
            bits.append(f"opposed by {', '.join(_unique(st['oppose']))}")
        if bits:
            parts.append(f"{o['title']}: " + "; ".join(bits))

    open_objections = sum(1 for o in objections
                          if o.get("resolution_status", "open") == "open" and not o.get("moot"))
    open_blockers = sum(1 for b in blockers if b.get("resolution_status", "open") == "open")
    unowned = sum(1 for a in actions if not _named_owner(a))

    parts.append(f"{open_objections} open objection{'s' if open_objections != 1 else ''}")
    parts.append(f"{open_blockers} open blocker{'s' if open_blockers != 1 else ''}")
    if unowned:
        parts.append(f"{unowned} action{'s' if unowned != 1 else ''} without a named owner")

    return "; ".join(parts) + "."


def _compute_preference(brief_body: dict) -> dict | None:
    """Directional lean, computed from positions only.

    An option qualifies only when at least two different participants gave an explicit
    "support" and no open (non-moot) objection targets it. The text only counts who
    said what; it never uses words like consensus or agreed.
    """
    objections = brief_body.get("objections", [])
    positions = brief_body.get("positions", [])
    lines: list[str] = []
    support_refs: list[str] = []
    dissent_refs: list[str] = []
    first_id = ""

    for o in brief_body.get("options", []):
        if o.get("status") == "dropped" or "id" not in o:
            continue
        ps = [p for p in positions if p.get("option_id") == o["id"]]
        supporters = [p for p in ps if p["stance"] == "support"]
        names = _unique(p["participant_label"] for p in supporters)
        if len(names) < 2:
            continue
        if any(o["id"] in ob.get("option_ids", []) and ob.get("resolution_status", "open") == "open"
               and not ob.get("moot") for ob in objections):
            continue
        line = f"{o['title']} has explicit support from {_join_names(names)}"
        cond = _unique(p["participant_label"] for p in ps if p["stance"] == "conditional")
        if cond:
            line += f" (conditional: {_join_names(cond)})"
        proposers = [n for n in _unique(p["participant_label"] for p in ps
                                        if p["stance"] == "proposed" or p.get("proposer")) if n not in names]
        if proposers:
            line += f"; proposed by {_join_names(proposers)}"
        ack = [n for n in _unique(p["participant_label"] for p in ps if p["stance"] == "acknowledged")
               if n not in names]
        if ack:
            line += f"; acknowledged only by {_join_names(ack)}"
        opp = _unique(p["participant_label"] for p in ps if p["stance"] == "oppose")
        if opp:
            line += f"; opposed by {_join_names(opp)}"
        lines.append(line + ".")
        support_refs += [r for p in supporters for r in p.get("source_unit_ids", [])]
        dissent_refs += [r for p in ps if p["stance"] == "oppose" for r in p.get("source_unit_ids", [])]
        first_id = first_id or o["id"]

    if not lines:
        return None
    return {"option_id": first_id, "summary": " ".join(lines),
            "supporting_refs": _unique(support_refs), "dissenting_refs": _unique(dissent_refs)}


# ---------------------------------------------------------------------------
# Exec summary — LLM phrasing + verification
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM = """\
You are the summary step of Pact, a facilitator for group decisions.
Produce a structured executive summary from the Decision State. You summarize; you do not decide, judge, or add information.

FIELD RULES
- title and status are filled in by the system. Put "-" in them.
- needs_attention: at most 5 items. Rank: open objections (skip moot ones), open blockers, actions whose owner_kind is not "named" (say "owner not named"), conflicting or unverified claims the proposals rely on. Use only items present in the state.
- proposed: each option with the exact support level taken from positions: "explicit support from A", "acknowledged by B", "conditional: C (condition)", "opposed by D". Never merge acknowledged into support. Mention the proposer only if a position says so. Dropped options get one short line saying they were dropped.
- committed: actions whose owner_kind is "named", with the due date if stated. Do not list a task as committed unless the quote shows the named owner committing to it or being asked to do it.
- unverified_claims: claims with evidence_status unverified, conflicting or unresolved.
- Items marked sensitive, and source messages tagged [SENSITIVE], may contain private health or financial details. State only the requirement, never the private reason.
- Email sources: being on To or Cc is not agreement. Never say or imply that someone responded, agreed, declined or stayed silent unless a quote shows it. A request addressed to a person means that person was asked, not that they accepted. A forwarding note ("FYI", "keep it to yourself") is the forwarder's remark, not a position on the forwarded text. Items carry `seen_by` (who saw every source) and `restricted` (the sender asked for privacy); never suggest sharing a restricted item more widely.
- Meeting sources: spoken "this one"/"that one" often points at a screen you cannot see; do not state what it referred to.

REQUIRED FOR EVERY ITEM in needs_attention, proposed, committed, unverified_claims:
  verbatim_quote  — exact consecutive words copied from the source message, MAX 15 WORDS.
                    Do NOT paraphrase, summarize, or alter even one word.
  source_unit_id  — the id of the source unit that contains that exact quote.

If you cannot find an exact quote in the source for an item, OMIT THAT ITEM ENTIRELY.
Never invent a quote. Never use placeholder ids.

BANNED WORDS (never use): consensus, agreed, decided, leans, converging, converged,
  effectively, eliminated.
BANNED PLACEHOLDERS (never use): (source), (cite), (id), [source].
"""


def _cited_item_schema() -> dict:
    return {
        "type": "object",
        "required": ["text", "verbatim_quote", "source_unit_id"],
        "properties": {
            "text": {"type": "string"},
            "verbatim_quote": {
                "type": "string",
                "description": "Exact consecutive words from the cited source unit, max 15 words",
            },
            "source_unit_id": {"type": "string"},
        },
    }


_VERIFIER_TOOL = {
    "name": "produce_exec_summary",
    "description": "Produce a structured executive summary with verbatim quotes per item.",
    "input_schema": {
        "type": "object",
        "required": ["title", "status", "needs_attention", "proposed",
                     "committed", "unverified_claims"],
        "properties": {
            "title": {"type": "string"},
            "status": {"type": "string"},
            "needs_attention": {"type": "array", "items": _cited_item_schema()},
            "proposed": {"type": "array", "items": _cited_item_schema()},
            "committed": {"type": "array", "items": _cited_item_schema()},
            "unverified_claims": {"type": "array", "items": _cited_item_schema()},
        },
    },
}

_VERIFIABLE_SECTIONS = ("needs_attention", "proposed", "committed", "unverified_claims")
_MAX_QUOTE_WORDS = 15


def _verify_items(
    items: list[dict], unit_text_map: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Return (verified, rejected).

    An item passes if its verbatim_quote (at most 15 words) appears in the cited source
    unit's text after normalising case, whitespace and curly quotes.
    """
    verified, rejected = [], []
    for item in items:
        quote = (item.get("verbatim_quote") or "").strip()
        source_text = unit_text_map.get(item.get("source_unit_id", ""), "")
        if (quote and source_text and len(quote.split()) <= _MAX_QUOTE_WORDS
                and quote_in_text(quote, source_text)):
            verified.append(item)
        else:
            rejected.append(item)
    return verified, rejected


_ENTAIL_SYSTEM = """\
You check whether a quoted chat message supports a summary line. For each item you get the summary TEXT, the QUOTE, the SPEAKER of the quote, and the SECTION it appears in.

Mark supported=true only if the quote directly states or clearly entails the text. Otherwise mark supported=false.
- committed: the quote must show the named owner committing to the task, or being asked to do it. A deadline from a third party, a statement of availability, or a report that the task is already done does NOT support a commitment.
- proposed: the support level in the text must match the quote. "ok", "fine", "sure whatever", "sounds good" support only "acknowledged", never "explicit support".
- needs_attention: the quote must show the problem the text describes (for example a task stated with no named owner, an open question, a dependency).
- unverified_claims: the quote must contain the claim, attributed to its speaker.
Be strict. When in doubt, supported=false.
"""

_ENTAIL_TOOL = {
    "name": "check_support",
    "description": "Return one verdict per item.",
    "input_schema": {
        "type": "object",
        "required": ["verdicts"],
        "properties": {
            "verdicts": {"type": "array", "items": {
                "type": "object", "required": ["id", "supported"],
                "properties": {
                    "id": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            }},
        },
    },
}


async def _apply_entailment(summary: dict, unit_meta: dict[str, dict], llm) -> tuple[dict, list[dict]]:
    """Drop items whose quote does not actually support their text. Missing verdicts count as unsupported."""
    payload = []
    for section in _VERIFIABLE_SECTIONS:
        for i, item in enumerate(summary.get(section, [])):
            meta = unit_meta.get(item.get("source_unit_id", ""), {})
            payload.append({"id": f"{section}:{i}", "section": section, "text": item["text"],
                            "quote": item["verbatim_quote"], "speaker": meta.get("author", "")})
    if not payload:
        return summary, []

    result = await llm.extract(_ENTAIL_SYSTEM, json.dumps(payload, indent=2), _ENTAIL_TOOL, timeout=60)
    ok = {v["id"]: bool(v.get("supported")) for v in result.get("verdicts", [])}

    rejected: list[dict] = []
    for section in _VERIFIABLE_SECTIONS:
        kept = []
        for i, item in enumerate(summary.get(section, [])):
            if ok.get(f"{section}:{i}", False):
                kept.append(item)
            else:
                rejected.append(item)
        summary[section] = kept
    return summary, rejected


def _ref_label(uid: str, unit_meta: dict[str, dict] | None) -> str:
    meta = (unit_meta or {}).get(uid)
    if meta and (meta.get("author") or meta.get("ts")):
        return ", ".join(x for x in (meta.get("author"), meta.get("ts")) if x)
    return f"m{uid[:6]}"


def _render_verified_summary(summary: dict, unit_meta: dict[str, dict] | None = None) -> str:
    """Render a verified summary dict to the plain-text format used by BriefTab."""
    lines: list[str] = []
    lines.append(summary.get("title", ""))
    lines.append("")
    lines.append(summary.get("status", ""))
    if summary.get("scope"):
        lines.append(summary["scope"])

    section_headings = {
        "needs_attention": "NEEDS ATTENTION",
        "proposed": "PROPOSED",
        "committed": "COMMITTED",
        "unverified_claims": "UNVERIFIED",
    }
    for key, heading in section_headings.items():
        items = summary.get(key, [])
        if not items:
            continue
        lines.append("")
        lines.append(heading)
        for item in items:
            uid = item.get("source_unit_id", "")
            ref = f" [{_ref_label(uid, unit_meta)}]" if uid else ""
            if summary.get("mark_restricted") and (unit_meta or {}).get(uid, {}).get("restricted"):
                ref += " [private]"
            lines.append(f"· {item['text']}{ref}")

    pref = (summary.get("preference") or "").strip()
    if pref:
        lines.append("")
        lines.append("PREFERENCE")
        lines.append(pref)

    return "\n".join(lines)


async def _generate_exec_summary(
    brief_body: dict,
    llm,
    unit_text_map: dict[str, str],
    unit_meta: dict[str, dict] | None = None,
    report: dict | None = None,
    scope: str = "",
) -> str:
    """Generate, verify and render the executive summary.

    Title, status and preference are computed in Python. Every LLM-written item must
    (1) quote its source verbatim and (2) pass an entailment check. Anything that fails
    is removed, never shipped. ``report`` (optional) receives ``verified``,
    ``omitted_items`` and ``issues`` so callers can store them.
    """
    unit_meta = unit_meta or {}
    issues: list[str] = []
    entailment_ran = True

    computed_status = _compute_status_line(brief_body)
    pref = brief_body.get("emerging_preference")
    pref_text = (pref or {}).get("summary") or "No directional preference recorded."
    question = (brief_body.get("question") or {}).get("text") or ""

    opt_title = {o["id"]: o["title"] for o in brief_body.get("options", []) if "id" in o}
    state = {k: brief_body.get(k) for k in (
        "question", "options", "criteria", "constraints", "claims",
        "positions", "objections", "blockers", "separate_issues", "actions",
    )}
    # Resolve option ids to titles so the model never has to join on a UUID.
    state["positions"] = [{**p, "option_title": opt_title.get(p.get("option_id", ""), "")}
                          for p in brief_body.get("positions", [])]
    state["objections"] = [{**o, "option_titles": [opt_title.get(i, "") for i in o.get("option_ids", [])]}
                           for o in brief_body.get("objections", [])]

    referenced_ids: set[str] = set()
    sensitive_ids: set[str] = set()
    for section in _BRIEF_SECTIONS:
        for item in brief_body.get(section, []):
            referenced_ids.update(item.get("source_unit_ids", []))
            if item.get("sensitive"):
                sensitive_ids.update(item.get("source_unit_ids", []))

    source_lines = "\n".join(
        f"{uid}: {unit_text_map[uid]}"
        + (" [SENSITIVE: state only the requirement, never the private reason]" if uid in sensitive_ids else "")
        for uid in sorted(referenced_ids)
        if uid in unit_text_map
    )
    source_block = f"\nSource Messages (id: text):\n{source_lines}\n" if source_lines else ""

    user_base = (
        f"Decision State:\n{json.dumps(state, indent=2)}{source_block}\n"
        f"Computed status (the system uses this exactly; put '-' in the status field):\n{computed_status}"
    )
    user = user_base

    rendered = ""
    omitted = 0
    attempts = 0
    for _attempt in range(2):
        attempts += 1
        summary = await llm.extract(_VERIFIER_SYSTEM, user, _VERIFIER_TOOL, timeout=60)

        # Deterministic fields: the LLM cannot override these.
        summary["status"] = computed_status
        summary["title"] = question or summary.get("title", "")
        summary["preference"] = pref_text
        summary["scope"] = scope
        cited = [u for u in cited_unit_ids(brief_body) if u in unit_meta]
        n_private = sum(1 for u in cited if unit_meta[u].get("restricted"))
        summary["mark_restricted"] = 0 < n_private < len(cited)

        rejected_quotes: list[dict] = []
        for section in _VERIFIABLE_SECTIONS:
            verified, rejected = _verify_items(summary.get(section, []), unit_text_map)
            summary[section] = verified
            rejected_quotes.extend(rejected)

        rejected_support: list[dict] = []
        try:
            summary, rejected_support = await _apply_entailment(summary, unit_meta, llm)
        except Exception as exc:  # noqa: BLE001 - never ship unchecked items silently
            entailment_ran = False
            issues.append(f"entailment check unavailable: {exc}")

        rendered = _render_verified_summary(summary, unit_meta)
        violations = _check_banned_words(rendered)
        omitted = len(rejected_quotes) + len(rejected_support)

        if not rejected_quotes and not rejected_support and not violations:
            break

        feedback: list[str] = []
        if rejected_quotes:
            bad = [f"'{r.get('verbatim_quote', '')}' (cited unit {r.get('source_unit_id', '')})"
                   for r in rejected_quotes]
            feedback.append(
                f"These quotes could not be found verbatim (max {_MAX_QUOTE_WORDS} words) in their cited "
                f"source units and were removed: {bad}. Only include items whose exact words appear in the source."
            )
        if rejected_support:
            feedback.append(
                "These items were removed because their quote does not support the stated text, owner "
                f"or support level: {[r.get('text', '') for r in rejected_support]}."
            )
        if violations:
            feedback.append(f"Banned words or placeholders found: {violations}. Rewrite without them.")
        user = user_base + "\n\n" + " ".join(feedback)

    # Fail closed: whatever still violates the rules is removed, not shipped.
    leftover = _check_banned_words(rendered)
    if leftover:
        rendered, removed = _strip_banned_lines(rendered)
        omitted += removed
        issues.append(f"removed {removed} line(s) containing banned words {leftover}")
    if omitted:
        issues.append(f"{omitted} item(s) omitted because they failed verification")

    if report is not None:
        report["verified"] = entailment_ran
        report["omitted_items"] = omitted
        report["attempts"] = attempts
        report["issues"] = issues
    return rendered


# ---------------------------------------------------------------------------
# Viewer-specific summaries
# ---------------------------------------------------------------------------

async def exec_summary_for_viewer(
    brief_body: dict,
    viewer: str,
    llm,
    unit_text_map: dict[str, str],
    unit_meta: dict[str, dict],
) -> dict:
    """Executive summary containing only what ``viewer`` could see.

    Items whose sources the viewer was not on are removed, partly hidden items lose the hidden
    sources, and the status line and preference are recomputed from what is left. A viewer who
    can see nothing gets an empty summary rather than a guess.
    """
    visible, stats = filter_for_viewer(brief_body, unit_meta, viewer)
    visible["emerging_preference"] = _compute_preference(visible)
    visible.pop("executive_summary", None)
    if not any(visible.get(sec) for sec in _BRIEF_SECTIONS):
        return {"summary": "", **stats, "note": "nothing in this thread was visible to this person"}
    report: dict = {}
    text = await _generate_exec_summary(
        visible, llm, unit_text_map, unit_meta=unit_meta, report=report,
        scope=scope_line(unit_meta, cited_unit_ids(visible)),
    )
    return {"summary": text, **stats, "report": report}


# ---------------------------------------------------------------------------
# Server-side id assignment and reference validation
# ---------------------------------------------------------------------------

def _resolve_option_id(title: str, options: list[dict], by_title: dict[str, str]) -> str:
    t = (title or "").lower()
    if t in by_title:
        return by_title[t]
    match = next((o for o in options if _same_option(title or "", o.get("title", ""))), None)
    return match["id"] if match else ""


def _assign_server_ids(brief_body: dict) -> dict:
    """Replace any LLM-generated IDs with server-assigned UUIDs.

    Also resolves positions.option_id from option_title (exact, then fuzzy) so
    cross-references remain consistent after ID assignment.
    """
    option_title_to_id: dict[str, str] = {}
    for item in brief_body.get("options", []):
        item.pop("id", None)
        new_id = str(uuid.uuid4())
        item["id"] = new_id
        option_title_to_id[item["title"].lower()] = new_id

    for section in _BRIEF_SECTIONS:
        if section == "options":
            continue
        for item in brief_body.get(section, []):
            item.pop("id", None)
            item["id"] = str(uuid.uuid4())

    options = brief_body.get("options", [])
    for pos in brief_body.get("positions", []):
        title = pos.pop("option_title", "")
        pos["option_id"] = _resolve_option_id(title, options, option_title_to_id)

    for obj in brief_body.get("objections", []):
        titles = obj.pop("option_titles", [])
        ids = [_resolve_option_id(t, options, option_title_to_id) for t in titles]
        obj["option_ids"] = [i for i in ids if i]

    return brief_body


async def _validate_source_refs(brief_body: dict, workspace_id: uuid.UUID, session: AsyncSession) -> list[str]:
    """Strip source_unit_ids that don't exist in this workspace. Returns warning strings."""
    result = await session.execute(
        select(SourceUnit.id).where(SourceUnit.workspace_id == workspace_id)
    )
    valid_ids = {str(r) for r in result.scalars().all()}

    warnings: list[str] = []
    for section in _BRIEF_SECTIONS:
        for item in brief_body.get(section, []):
            original = item.get("source_unit_ids", [])
            filtered = [sid for sid in original if sid in valid_ids]
            if len(filtered) < len(original):
                bad = len(original) - len(filtered)
                warnings.append(
                    f"{section}[{item.get('id', '?')}]: dropped {bad} invalid source_unit_id(s)"
                )
                item["source_unit_ids"] = filtered
    return warnings


async def run_synthesize_brief(payload: dict, session: AsyncSession) -> dict:
    from .llm import LLMClient

    workspace_id = uuid.UUID(payload["workspace_id"])
    extract_job_id = uuid.UUID(payload["extract_job_id"])

    workspace = await session.get(Workspace, workspace_id)
    if workspace is None:
        raise ValueError(f"Workspace {workspace_id} not found")

    extract_job = await session.get(Job, extract_job_id)
    if extract_job is None or extract_job.status != JobStatus.done:
        raise ValueError(f"extract_brief job {extract_job_id} is not done")

    merged = extract_job.output_payload.get("merged_draft", {})

    # Use confirmed question if set, otherwise fall back to top extracted candidate
    primary_question = workspace.primary_question
    if not primary_question:
        candidates = merged.get("candidate_questions", [])
        primary_question = candidates[0]["text"] if candidates else None

    # Build user prompt from merged candidates
    candidates_json = json.dumps({
        k: merged.get(k, [])
        for k in ("options", "criteria", "constraints", "claims",
                  "positions", "objections", "blockers", "separate_issues", "actions")
    }, indent=2)

    question_line = f"Decision question: {primary_question}" if primary_question else "Decision question: not specified"
    user = (
        f"{question_line}\n"
        f"Scope: {workspace.scope or 'Not specified'}\n\n"
        f"Extracted candidates from the full discussion:\n{candidates_json}\n\n"
        "Synthesize a structured decision brief."
    )

    llm = LLMClient()
    brief_body = await llm.extract(_SYNTH_SYSTEM, user, _SYNTH_TOOL, timeout=120)

    # The synthesis model may not drop items or rewrite stances, owners or statuses.
    restore_notes = _restore_dropped(brief_body, merged)

    # Server assigns all IDs (LLM must not generate them)
    brief_body = _assign_server_ids(brief_body)

    # Validate and strip any source_unit_ids that don't belong to this workspace
    ref_warnings = await _validate_source_refs(brief_body, workspace_id, session)

    # Add the question metadata
    brief_body["question"] = {
        "text": primary_question,
        "scope": workspace.scope,
        "sub_decisions": sorted({o["decision"] for o in brief_body.get("options", []) if o.get("decision")}),
    }

    # The directional lean is computed from positions, never written by the model.
    brief_body["emerging_preference"] = _compute_preference(brief_body)

    # Unit text and metadata for the verifier pass and readable references
    units_result = await session.execute(
        select(SourceUnit.id, SourceUnit.normalized_text, SourceUnit.author_label, SourceUnit.locator)
        .where(SourceUnit.workspace_id == workspace_id)
    )
    unit_text_map: dict[str, str] = {}
    unit_meta: dict[str, dict] = {}
    for row in units_result:
        uid = str(row.id)
        unit_text_map[uid] = row.normalized_text or ""
        loc = row.locator or {}
        unit_meta[uid] = {
            "author": row.author_label or "", "ts": loc.get("timestamp_str", ""),
            "effective_audience": loc.get("effective_audience"),
            "audience_unknown": bool(loc.get("audience_unknown")),
            "restricted": bool(loc.get("restricted")),
        }

    # Who saw what is computed from the cited messages, never by the model.
    attach_audience(brief_body, unit_meta)
    scope = scope_line(unit_meta, cited_unit_ids(brief_body))

    # Generate exec summary — verifier pass strips items with unverifiable quotes
    report: dict = {}
    brief_body["executive_summary"] = await _generate_exec_summary(
        brief_body, llm, unit_text_map, unit_meta=unit_meta, report=report, scope=scope
    )
    brief_body["executive_summary_report"] = report

    # Optional: one summary per named viewer, built only from what that person could see.
    viewers = payload.get("viewers") or []
    if viewers:
        brief_body["executive_summary_by_viewer"] = {}
        for viewer in viewers:
            brief_body["executive_summary_by_viewer"][viewer] = await exec_summary_for_viewer(
                brief_body, viewer, llm, unit_text_map, unit_meta
            )

    # Re-acquire workspace lock after the LLM call so current_revision_id is fresh.
    # Without this, a concurrent accept/review that committed between the initial
    # session.get() and here would leave us writing with a stale parent pointer.
    locked_ws = await session.execute(
        select(Workspace)
        .where(Workspace.id == workspace_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    workspace = locked_ws.scalar_one()

    # Compute next revision number
    current_rev_num = 0
    if workspace.current_revision_id:
        current_rev = await session.get(BriefRevision, workspace.current_revision_id)
        if current_rev:
            current_rev_num = current_rev.revision_number

    event = Event(
        workspace_id=workspace_id,
        event_type="brief_synthesized",
        actor_type="ai",
        actor_label="pact-extractor",
        payload={
            "unit_count": extract_job.output_payload.get("unit_count", 0),
            "ref_warnings": ref_warnings,
            "restored_or_corrected": restore_notes,
            "dropped_at_extraction": extract_job.output_payload.get("dropped_item_count", 0),
            "executive_summary_report": report,
        },
        prior_revision=current_rev_num,
        resulting_revision=current_rev_num + 1,
    )
    session.add(event)
    await session.flush()

    revision = BriefRevision(
        workspace_id=workspace_id,
        revision_number=current_rev_num + 1,
        parent_revision_id=workspace.current_revision_id,
        structured_snapshot=brief_body,
        event_id=event.id,
    )
    session.add(revision)
    await session.flush()

    workspace.current_revision_id = revision.id
    await session.commit()

    return {
        "revision_id": str(revision.id),
        "revision_number": revision.revision_number,
    }
