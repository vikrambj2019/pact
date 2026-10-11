"""Staged extraction orchestration — runs extract_brief and synthesize_brief jobs."""
from __future__ import annotations

import copy
import json
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.models import (
    BriefRevision, Event, Job, JobStatus, JobType, SourceUnit, Workspace,
)
from .extractor import quote_in_text
from .merger import _norm_text, _same_option, _similar


# ---------------------------------------------------------------------------
# Job 1: extract_brief  (chunk → per-chunk extract → merge → candidate questions)
# ---------------------------------------------------------------------------

async def extract_and_merge(units_data: list[dict], llm) -> tuple[dict, int]:
    """Chunk → per-chunk extraction → merge. No database access; returns (merged, chunk_count).

    ``units_data``: dicts with id, sequence, author_label, normalized_text, locator.
    """
    from .chunker import chunk_units
    from .extractor import extract_chunk
    from .merger import merge_extractions

    unit_sequence = {u["id"]: u["sequence"] for u in units_data}
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
        for item in extracted.get("actions", []):
            key = _norm_text(item["text"])
            if item.get("status", "open") == "open":
                open_items[key] = {"kind": "action", "text": item["text"]}
            else:
                open_items.pop(key, None)

    return merge_extractions(chunk_results, unit_sequence=unit_sequence), len(chunks)


async def run_extract_brief(payload: dict, session: AsyncSession) -> dict:
    from .llm import LLMClient

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

    llm = LLMClient()
    merged, chunk_count = await extract_and_merge(units_data, llm)

    return {
        "candidate_questions": merged["candidate_questions"],
        "merged_draft": merged,
        "unit_count": len(units_data),
        "chunk_count": chunk_count,
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
- Keep each option's `decision` and `status`, each position's `stance`, each action's `owner_kind`, `owner_label`, `due` and `status`, and each objection's and blocker's `resolution_status` exactly as given. Never turn "acknowledged" into support. Never assign an owner the candidates do not name.
- Keep `sensitive` flags. For sensitive items state only the requirement, never the private reason.
- For claims: evidence_status is "supported_within_scope" only when another participant gives evidence or a reason; a bare assertion is "unverified", including one person's report of what a venue told them ("ground floor", "free parking", "about $20 a head"); use "conflicting" when participants give different values for the same thing; use "contradicted_within_scope" when a later message corrects the figure ("that's the a la carte number, prix fixe is $95"), when the person a claim is about disputes it ("Yusuf is fine with anything" / "I care"), or when a claim that everyone agrees is contradicted by any participant or by missing replies.
- Keep a dropped option's `status_reason`.
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
                    "status": {"type": "string", "enum": ["open", "done", "declined"]},
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

def _pair_items(items: list[dict], cands: list[dict], same) -> tuple[list[tuple[dict, dict | None]], list[dict]]:
    """Pair each merged candidate with at most one synthesized item.

    Pass 1 matches by text (``same``). Pass 2 catches rephrasings: an unmatched item that
    cites exactly the same source messages as an unmatched candidate. Returns
    ([(candidate, item or None)], unmatched_items).
    """
    used: set[int] = set()
    pairs: list[list] = []
    for cand in cands:
        idx = next((i for i, it in enumerate(items) if i not in used and same(it, cand)), None)
        if idx is not None:
            used.add(idx)
        pairs.append([cand, idx])
    for pair in pairs:
        if pair[1] is not None:
            continue
        ids = set(pair[0].get("source_unit_ids") or [])
        idx = next((i for i, it in enumerate(items)
                    if i not in used and ids and set(it.get("source_unit_ids") or []) == ids), None)
        if idx is not None:
            used.add(idx)
            pair[1] = idx
    return ([(c, items[i] if i is not None else None) for c, i in pairs],
            [it for i, it in enumerate(items) if i not in used])


def _restore_dropped(brief_body: dict, merged: dict) -> list[str]:
    """Make the identity-bearing sections of the brief agree with the merged draft.

    Options, positions, objections, blockers and actions feed the status counts and the
    preference line, so the synthesis model may reword them but may not add, drop or
    re-score them:

    * positions are taken from the merged draft verbatim (they are fully deterministic);
    * for the other sections each merged item is paired with a synthesized one (by text,
      then by identical source messages); missing items are restored, fields the model
      must not change are reset, and synthesized items with no merged counterpart are
      removed, because nothing in the extraction supports them.

    Run BEFORE ``_assign_server_ids`` (titles still present). Returns notes.
    """
    notes: list[str] = []

    # options
    opts = brief_body.get("options", []) or []
    pairs, extra = _pair_items(opts, merged.get("options", []),
                               lambda it, c: _same_option(it.get("title", ""), c["title"],
                                                          it.get("decision", ""), c.get("decision", "")))
    new_opts = []
    for cand, match in pairs:
        if match is None:
            match = {**{k: cand[k] for k in ("title", "description", "decision", "status", "status_reason", "label",
                                             "source_unit_ids") if k in cand},
                     "restored": True}
            notes.append(f"restored option '{cand['title']}'")
        for f in ("decision", "status"):
            if cand.get(f) and match.get(f) != cand[f]:
                notes.append(f"option '{match.get('title')}': {f} {match.get(f)!r} -> {cand[f]!r}")
                match[f] = cand[f]
        for f in ("status_reason", "label"):
            if cand.get(f):
                match[f] = cand[f]
            else:
                match.pop(f, None)
        match["title"] = cand["title"]          # positions and objections reference this title
        new_opts.append(match)
    notes += [f"removed unsupported option '{o.get('title', '')}'" for o in extra]
    brief_body["options"] = new_opts

    # positions: deterministic, straight from the merge
    llm_positions = brief_body.get("positions", []) or []
    keep = ("participant_label", "option_title", "stance", "conditions", "proposer", "source_unit_ids")
    brief_body["positions"] = [{k: p[k] for k in keep if k in p} for p in merged.get("positions", [])]
    if len(llm_positions) != len(brief_body["positions"]):
        notes.append(f"positions taken from merge ({len(brief_body['positions'])}); "
                     f"synthesis returned {len(llm_positions)}")

    # objections and blockers: status comes from chronology, not from the LLM
    for section in ("objections", "blockers"):
        pairs, extra = _pair_items(brief_body.get(section, []) or [], merged.get(section, []),
                                   lambda it, c: _similar(it.get("text", ""), c["text"]))
        out = []
        for cand, match in pairs:
            if match is None:
                keep = ("text", "kind", "option_titles", "author_label", "moot", "sensitive",
                        "resolution_status", "source_unit_ids")
                match = {**{k: cand[k] for k in keep if k in cand}, "restored": True}
                notes.append(f"restored {section[:-1]} '{cand['text'][:60]}'")
            if cand.get("resolution_status") and match.get("resolution_status") != cand["resolution_status"]:
                notes.append(f"{section[:-1]} '{match['text'][:50]}': status "
                             f"{match.get('resolution_status')!r} -> {cand['resolution_status']!r}")
                match["resolution_status"] = cand["resolution_status"]
            if cand.get("option_titles"):
                match["option_titles"] = cand["option_titles"]
            match["moot"] = bool(cand.get("moot"))
            out.append(match)
        notes += [f"removed unsupported {section[:-1]} '{i.get('text', '')[:60]}'" for i in extra]
        brief_body[section] = out

    # actions: owner, due and status are whatever the chat said, never an LLM guess
    pairs, extra = _pair_items(brief_body.get("actions", []) or [], merged.get("actions", []),
                               lambda it, c: _similar(it.get("text", ""), c["text"], 0.75))
    out = []
    for cand, match in pairs:
        if match is None:
            keep = ("text", "owner_label", "owner_kind", "due", "status", "acceptance", "deadline",
                    "deadline_source_unit_ids", "sensitive", "source_unit_ids")
            match = {**{k: cand[k] for k in keep if k in cand}, "restored": True}
            notes.append(f"restored action '{cand['text'][:60]}'")
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
        else:
            match.pop("due", None)
        status = cand.get("status") or "open"
        if match.get("status", "open") != status:
            notes.append(f"action '{match['text'][:50]}': status {match.get('status')!r} -> {status!r}")
        match["status"] = status
        acceptance = cand.get("acceptance") or "committed"
        if match.get("acceptance", acceptance) != acceptance:
            notes.append(f"action '{match['text'][:50]}': acceptance {match.get('acceptance')!r} -> {acceptance!r}")
        match["acceptance"] = acceptance
        for f in ("deadline", "deadline_source_unit_ids"):
            if cand.get(f):
                match[f] = cand[f]
            else:
                match.pop(f, None)
        out.append(match)
    notes += [f"removed unsupported action '{a.get('text', '')[:60]}'" for a in extra]
    brief_body["actions"] = out

    # sensitivity flags are sticky in every section
    for section in ("constraints", "claims", "objections", "blockers", "actions"):
        for cand in merged.get(section, []):
            if cand.get("sensitive"):
                for item in brief_body.get(section, []):
                    if _similar(item.get("text", ""), cand["text"]) or (
                            set(item.get("source_unit_ids") or []) & set(cand.get("source_unit_ids") or [])):
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


_NEGATION_RE = re.compile(r"\b(?:not|no|never|nothing|nobody|none|without|hasn't|haven't|hadn't|isn't|"
                          r"wasn't|weren't|didn't|doesn't|don't|hasnt|havent|didnt|yet)\b|n't\b", re.IGNORECASE)


def _negated(text: str, start: int) -> bool:
    """True when one of the four words before ``start`` negates it ("hasn't agreed",
    "no due date agreed"): a negated claim of agreement is exactly what the card should say."""
    window = " ".join(text[:start].split()[-4:])
    return bool(_NEGATION_RE.search(window))


def _check_banned_words(text: str) -> list[str]:
    """Whole-word match, so 'disagreed', 'undecided' and 'cleans' are not flagged; negated
    uses ("hasn't agreed", "not decided") are allowed."""
    found = sorted({m.group(1).lower() for m in _BANNED_RE.finditer(text) if not _negated(text, m.start())})
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


def _action_open(action: dict) -> bool:
    return action.get("status", "open") == "open"


def _affirmers(ps: list[dict]) -> list[str]:
    """Explicit yes: a support, or the proposer's own still-standing proposal."""
    return _unique(p["participant_label"] for p in ps
                   if p["stance"] == "support" or p["stance"] == "proposed")


def _option_name(o: dict) -> str:
    return f"{o['label']}) {o['title']}" if o.get("label") else o["title"]


def _support_breakdown(ps: list[dict]) -> dict[str, list[str]]:
    """Who said what about one option, each person counted once at their strongest stance."""
    support = _unique(p["participant_label"] for p in ps if p["stance"] == "support")
    cond = [n for n in _unique(p["participant_label"] for p in ps if p["stance"] == "conditional") if n not in support]
    taken = set(support) | set(cond)
    proposer = [n for n in _unique(p["participant_label"] for p in ps
                                   if p["stance"] == "proposed" or p.get("proposer")) if n not in taken]
    taken |= set(proposer)
    ack = [n for n in _unique(p["participant_label"] for p in ps if p["stance"] == "acknowledged") if n not in taken]
    opp = _unique(p["participant_label"] for p in ps if p["stance"] == "oppose")
    return {"support": support, "conditional": cond, "proposer": proposer, "acknowledged": ack, "oppose": opp}


def _supporters_phrase(b: dict[str, list[str]]) -> str:
    """'3 explicit supporters (1 conditional): A, B, C (conditional)'."""
    names = b["support"] + [f"{n} (conditional)" for n in b["conditional"]]
    if not names:
        return ""
    n = len(names)
    head = f"{n} explicit supporter{'s' if n != 1 else ''}"
    if b["conditional"]:
        head += f" ({len(b['conditional'])} conditional)"
    return f"{head}: {', '.join(names)}"


def _compute_status_line(brief_body: dict) -> str:
    """Compute the status counts in Python so the LLM cannot get them wrong."""
    options = brief_body.get("options", [])
    positions = brief_body.get("positions", [])
    blockers = brief_body.get("blockers", [])
    objections = brief_body.get("objections", [])
    actions = brief_body.get("actions", [])
    participants = brief_body.get("participants") or []

    opt_by_id = {o["id"]: o for o in options if "id" in o}
    active = [o for o in options if o.get("status") != "dropped"]
    dropped = [o for o in options if o.get("status") == "dropped"]

    by_opt: dict[str, list[dict]] = {}
    for p in positions:
        o = opt_by_id.get(p.get("option_id", ""))
        if o is not None:
            by_opt.setdefault(o["id"], []).append(p)

    head = f"{len(options)} option{'s' if len(options) != 1 else ''} proposed"
    if dropped:
        head += f" ({len(dropped)} dropped: " + ", ".join(
            _option_name(o) + (f" — {o['status_reason']}" if o.get("status_reason") else "") for o in dropped) + ")"
    parts: list[str] = [head]

    for o in active:
        ps = by_opt.get(o.get("id", ""), [])
        b = _support_breakdown(ps)
        bits = []
        if o.get("status") == "fallback":
            bits.append("fallback" + (f" ({o['status_reason']})" if o.get("status_reason") else ""))
        if _supporters_phrase(b):
            bits.append(_supporters_phrase(b))
        if b["proposer"]:
            bits.append(f"proposed by {', '.join(b['proposer'])}")
        if b["acknowledged"]:
            bits.append(f"acknowledged only: {', '.join(b['acknowledged'])}")
        if b["oppose"]:
            bits.append(f"opposed by {', '.join(b['oppose'])}")
        if (b["support"] or b["conditional"] or b["proposer"]) and participants and o.get("decision"):
            # Silence is not agreement. Only live options count: opposing a dropped
            # option says nothing about the live choice.
            spoke = {p["participant_label"].lower() for q in active
                     if q.get("decision") == o.get("decision") for p in by_opt.get(q.get("id", ""), [])}
            silent = [n for n in participants if n.lower() not in spoke]
            if silent:
                bits.append(f"no stance on {o['decision']}: {', '.join(silent)}")
        if bits:
            label = f"{o['decision']} — {_option_name(o)}" if o.get("decision") else _option_name(o)
            parts.append(f"{label}: " + "; ".join(bits))

    open_obj = [o for o in objections if o.get("resolution_status", "open") == "open" and not o.get("moot")]
    targets = _unique(opt_by_id[i].get("label") or opt_by_id[i]["title"]
                      for ob in open_obj for i in ob.get("option_ids", []) if i in opt_by_id)
    open_blockers = sum(1 for b in blockers if b.get("resolution_status", "open") == "open")
    unowned = sum(1 for a in actions if _action_open(a) and not _named_owner(a))
    unaccepted = [a for a in actions if _action_open(a) and _named_owner(a) and a.get("acceptance") == "requested"]
    declined = sum(1 for a in actions if a.get("status") == "declined")

    parts.append(f"{len(open_obj)} open objection{'s' if len(open_obj) != 1 else ''}"
                 + (f" (on {', '.join(targets)})" if targets else ""))
    parts.append(f"{open_blockers} open blocker{'s' if open_blockers != 1 else ''}")
    if unowned:
        parts.append(f"{unowned} open action{'s' if unowned != 1 else ''} without a named owner")
    if unaccepted:
        parts.append(f"{len(unaccepted)} task{'s' if len(unaccepted) != 1 else ''} asked but not accepted ("
                     + ", ".join(a["owner_label"] for a in unaccepted) + ")")
    if declined:
        parts.append(f"{declined} assignment{'s' if declined != 1 else ''} declined by the person named")

    return "; ".join(parts) + "."


def _compute_preference(brief_body: dict) -> dict | None:
    """Directional lean, computed from positions only.

    An option qualifies only when at least two different participants gave an explicit
    yes (support, or their own standing proposal) and no open (non-moot) objection
    targets it. The text names who said what; it never uses words like consensus or agreed.
    """
    objections = brief_body.get("objections", [])
    positions = brief_body.get("positions", [])
    lines: list[str] = []
    support_refs: list[str] = []
    dissent_refs: list[str] = []
    first_id = ""

    for o in brief_body.get("options", []):
        if o.get("status") in ("dropped", "fallback") or "id" not in o:
            continue
        ps = [p for p in positions if p.get("option_id") == o["id"]]
        if len(_affirmers(ps)) < 2:
            continue
        if any(o["id"] in ob.get("option_ids", []) and ob.get("resolution_status", "open") == "open"
               and not ob.get("moot") for ob in objections):
            continue
        b = _support_breakdown(ps)
        names = b["support"] + [f"{n} (conditional)" for n in b["conditional"]]
        line = f"{o['title']} has explicit support from {_join_names(names)}" if names \
            else f"{o['title']} is proposed"
        if b["proposer"]:
            line += f", plus {_join_names(b['proposer'])} (proposer)" if names else f" by {_join_names(b['proposer'])}"
        if b["acknowledged"]:
            line += f"; acknowledged only by {_join_names(b['acknowledged'])}"
        if b["oppose"]:
            line += f"; opposed by {_join_names(b['oppose'])}"
        lines.append(line + ".")
        support_refs += [r for p in ps if p["stance"] in ("support", "proposed", "conditional")
                         for r in p.get("source_unit_ids", [])]
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
- needs_attention: at most 5 items. Rank: open objections (skip moot ones), open blockers, open actions whose owner_kind is not "named" (say "owner not named"), open actions with acceptance "requested" (say "asked of <owner>, not accepted"), actions with status "declined" (say who declined it), conflicting or unverified claims the proposals rely on. Use only items present in the state. A disagreement listed in separate_issues is never a blocker.
- proposed: each option with the exact support level taken from positions: "explicit support from A", "acknowledged by B", "conditional: C (condition)", "opposed by D". A "proposed" stance is the proposer's own explicit choice and counts as explicit support. Never merge acknowledged into support, and never count a participant with no position. Dropped options get one short line saying they were dropped and their status_reason; fallback options say they are a fallback and the condition. A summary message that says the group is aligned is a claim, never evidence of support.
- committed: actions whose owner_kind is "named", status is "open" and acceptance is "committed", with the due date if stated and any external deadline ("Monday; deadline Oct 16"). Never list "done", "declined" or "requested" actions here. Do not list a task as committed unless the quote shows the named owner committing to it or being asked to do it; if the owner was only asked and never replied, write "asked of <owner>", not as if they agreed.
- unverified_claims: claims with evidence_status unverified, conflicting, contradicted_within_scope or unresolved. Say what makes each one doubtful ("one person's report", "corrected later to $95", "Yusuf disputed this"). A corrected figure is never stated as fact.
- Items marked sensitive, and source messages tagged [SENSITIVE], may contain private health or financial details. State only the requirement, never the private reason.

REQUIRED FOR EVERY ITEM in needs_attention, proposed, committed, unverified_claims:
  verbatim_quote  — exact consecutive words copied from the source message, MAX 15 WORDS.
                    Do NOT paraphrase, summarize, or alter even one word.
  source_unit_id  — the id of the source unit that contains that exact quote.

If you cannot find an exact quote in the source for an item, OMIT THAT ITEM ENTIRELY.
Never invent a quote. Never use placeholder ids.

BANNED WORDS (never use as a claim): consensus, agreed, decided, leans, converging, converged,
  effectively, eliminated. Negated uses are fine and encouraged where true ("Dev hasn't agreed to Nov 6",
  "no due date agreed").
BANNED PLACEHOLDERS (never use): (source), (cite), (id), [source].
"""


_PRIVATE_SECTIONS = ("claims", "objections", "blockers", "actions")


def _summary_state(brief_body: dict) -> dict:
    """The part of the brief the card may draw on.

    Sensitive constraints stay (the card states the requirement, e.g. step-free access).
    Other sensitive items (a private money arrangement, a personal circumstance) never
    reach the card, so they are removed before the summary model sees the state.
    """
    state = {k: copy.deepcopy(brief_body.get(k)) for k in (
        "question", "options", "criteria", "constraints", "claims",
        "positions", "objections", "blockers", "separate_issues", "actions",
    )}
    for section in _PRIVATE_SECTIONS:
        state[section] = [i for i in (state.get(section) or []) if not i.get("sensitive")]
    for c in state.get("constraints") or []:
        if c.get("sensitive"):
            c.pop("quote", None)            # the quote can carry the private reason; the text cannot
            c.pop("basis", None)
    return state


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
- committed: the quote must show the named owner committing to the task, or being asked to do it. A deadline from a third party, a statement of availability, a report that the task is already done, or a quote where the owner refuses the task does NOT support a commitment. If the quote only shows someone else asking, the text must say the owner was asked, not that they committed.
- proposed: the support level in the text must match the quote. "ok", "fine", "sure whatever", "sounds good" support only "acknowledged", never "explicit support".
- needs_attention: the quote must show the problem the text describes (for example a task stated with no named owner, an open question, a dependency).
- unverified_claims: the quote must contain the claim, attributed to its speaker.
- Any section: when sensitive_source is true, supported=false if the text mentions a health detail, someone's money situation, or a private arrangement between people. Stating only a requirement ("step-free access required") is fine.
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


async def _apply_entailment(summary: dict, unit_meta: dict[str, dict], llm,
                            sensitive_ids: set[str] | None = None) -> tuple[dict, list[dict]]:
    """Drop items whose quote does not actually support their text. Missing verdicts count as unsupported."""
    sensitive_ids = sensitive_ids or set()
    payload = []
    for section in _VERIFIABLE_SECTIONS:
        for i, item in enumerate(summary.get(section, [])):
            meta = unit_meta.get(item.get("source_unit_id", ""), {})
            payload.append({"id": f"{section}:{i}", "section": section, "text": item["text"],
                            "quote": item["verbatim_quote"], "speaker": meta.get("author", ""),
                            "sensitive_source": item.get("source_unit_id", "") in sensitive_ids})
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
    state = _summary_state(brief_body)
    # Resolve option ids to titles so the model never has to join on a UUID.
    state["positions"] = [{**p, "option_title": opt_title.get(p.get("option_id", ""), "")}
                          for p in state.get("positions", [])]
    state["objections"] = [{**o, "option_titles": [opt_title.get(i, "") for i in o.get("option_ids", [])]}
                           for o in state.get("objections", [])]

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

        rejected_quotes: list[dict] = []
        for section in _VERIFIABLE_SECTIONS:
            verified, rejected = _verify_items(summary.get(section, []), unit_text_map)
            summary[section] = verified
            rejected_quotes.extend(rejected)

        rejected_support: list[dict] = []
        try:
            summary, rejected_support = await _apply_entailment(summary, unit_meta, llm, sensitive_ids)
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


def _strip_unknown_refs(brief_body: dict, valid_ids: set[str]) -> list[str]:
    """Strip source_unit_ids that are not in ``valid_ids``. Returns warning strings."""
    warnings: list[str] = []
    for section in _BRIEF_SECTIONS:
        for item in brief_body.get(section, []):
            original = item.get("source_unit_ids", [])
            filtered = [sid for sid in original if sid in valid_ids]
            if len(filtered) < len(original):
                warnings.append(f"{section}[{item.get('id', '?')}]: dropped "
                                f"{len(original) - len(filtered)} invalid source_unit_id(s)")
                item["source_unit_ids"] = filtered
    return warnings


async def build_brief(
    merged: dict,
    primary_question: str | None,
    scope: str | None,
    unit_text_map: dict[str, str],
    unit_meta: dict[str, dict],
    participants: list[str],
    llm,
) -> tuple[dict, list[str], list[str], dict]:
    """Merged draft → brief body with executive summary. No database access.

    Returns (brief_body, restore_notes, ref_warnings, summary_report).
    """
    candidates_json = json.dumps({
        k: merged.get(k, [])
        for k in ("options", "criteria", "constraints", "claims",
                  "positions", "objections", "blockers", "separate_issues", "actions")
    }, indent=2)
    question_line = f"Decision question: {primary_question}" if primary_question else "Decision question: not specified"
    user = (
        f"{question_line}\n"
        f"Scope: {scope or 'Not specified'}\n\n"
        f"Extracted candidates from the full discussion:\n{candidates_json}\n\n"
        "Synthesize a structured decision brief."
    )
    brief_body = await llm.extract(_SYNTH_SYSTEM, user, _SYNTH_TOOL, timeout=180, max_tokens=16000)

    # The synthesis model may not add, drop or re-score identity-bearing items.
    restore_notes = _restore_dropped(brief_body, merged)
    # Server assigns all IDs (LLM must not generate them)
    brief_body = _assign_server_ids(brief_body)
    ref_warnings = _strip_unknown_refs(brief_body, set(unit_text_map))

    brief_body["question"] = {
        "text": primary_question,
        "scope": scope,
        "sub_decisions": sorted({o["decision"] for o in brief_body.get("options", []) if o.get("decision")}),
    }
    # Everyone who wrote in the source, so the status line can name who said nothing.
    brief_body["participants"] = list(participants)
    # The directional lean is computed from positions, never written by the model.
    brief_body["emerging_preference"] = _compute_preference(brief_body)

    report: dict = {}
    brief_body["executive_summary"] = await _generate_exec_summary(
        brief_body, llm, unit_text_map, unit_meta=unit_meta, report=report
    )
    brief_body["executive_summary_report"] = report
    return brief_body, restore_notes, ref_warnings, report


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

    # Unit text and metadata for the verifier pass and readable references
    units_result = await session.execute(
        select(SourceUnit.id, SourceUnit.normalized_text, SourceUnit.author_label, SourceUnit.locator)
        .where(SourceUnit.workspace_id == workspace_id)
        .order_by(SourceUnit.sequence)
    )
    unit_text_map: dict[str, str] = {}
    unit_meta: dict[str, dict] = {}
    authors: list[str] = []
    for row in units_result:
        uid = str(row.id)
        unit_text_map[uid] = row.normalized_text or ""
        unit_meta[uid] = {"author": row.author_label or "", "ts": (row.locator or {}).get("timestamp_str", "")}
        if row.author_label:
            authors.append(row.author_label)

    brief_body, restore_notes, ref_warnings, report = await build_brief(
        merged, primary_question, workspace.scope, unit_text_map, unit_meta, _unique(authors), LLMClient(),
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
