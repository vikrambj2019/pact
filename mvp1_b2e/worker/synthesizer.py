"""Staged extraction orchestration — runs extract_brief and synthesize_brief jobs."""
from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.models import (
    BriefRevision, Event, Job, JobStatus, JobType, SourceUnit, Workspace,
)


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

    llm = LLMClient()
    chunks = chunk_units(units_data)
    chunk_results = []
    for chunk in chunks:
        extracted = await extract_chunk(chunk, llm)
        chunk_results.append(extracted)

    merged = merge_extractions(chunk_results)

    return {
        "candidate_questions": merged["candidate_questions"],
        "merged_draft": merged,
        "unit_count": len(units_data),
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------------------
# Job 2: synthesize_brief  (confirmed question + merged draft → BriefRevision)
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """\
You are a decision brief synthesizer. Given a set of extraction candidates from a discussion and a confirmed decision question, produce a clean, structured decision brief.

Rules:
- Only include items directly relevant to the confirmed decision question.
- Items not relevant to the confirmed question should go in separate_issues.
- Do NOT include an "id" field on any item — IDs are assigned by the server.
- All source_unit_ids you cite must come from the candidates provided — never invent IDs.
- For positions: only include if a participant explicitly stated their stance.
- emerging_preference: include only if the discussion shows a clear directional lean. Never label it consensus. Set to null if unclear.
- Keep all text concise and grounded in the source material.
"""

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
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "options": {"type": "array", "items": {
                "type": "object", "required": ["title", "source_unit_ids"],
                "properties": {
                    "title": {"type": "string"}, "description": {"type": "string"},
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
                               "enum": ["support", "oppose", "conditional", "undecided", "unknown"]},
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
                "type": "object", "required": ["text", "source_unit_ids"],
                "properties": {
                    "text": {"type": "string"}, "owner_label": {"type": "string"},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "emerging_preference": {
                "type": ["object", "null"],
                "properties": {
                    "option_title": {"type": "string"},
                    "summary": {"type": "string"},
                    "supporting_refs": {"type": "array", "items": {"type": "string"}},
                    "dissenting_refs": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }
}

_BRIEF_SECTIONS = ("criteria", "constraints", "options", "claims",
                   "positions", "objections", "blockers", "separate_issues", "actions")


def _assign_server_ids(brief_body: dict) -> dict:
    """Replace any LLM-generated IDs with server-assigned UUIDs.

    Also resolves positions.option_id from option_title so cross-references
    remain consistent after ID assignment.
    """
    # Build option title -> new server ID map (options processed first)
    option_title_to_id: dict[str, str] = {}
    for item in brief_body.get("options", []):
        item.pop("id", None)
        new_id = str(uuid.uuid4())
        item["id"] = new_id
        option_title_to_id[item["title"].lower()] = new_id

    # Assign IDs to all other sections
    for section in _BRIEF_SECTIONS:
        if section == "options":
            continue
        for item in brief_body.get(section, []):
            item.pop("id", None)
            item["id"] = str(uuid.uuid4())

    # Resolve positions: option_title -> option_id
    for pos in brief_body.get("positions", []):
        title = pos.pop("option_title", "").lower()
        pos["option_id"] = option_title_to_id.get(title, "")

    # Resolve objections: option_titles -> option_ids
    for obj in brief_body.get("objections", []):
        titles = obj.pop("option_titles", [])
        obj["option_ids"] = [option_title_to_id[t.lower()] for t in titles if t.lower() in option_title_to_id]

    # Resolve emerging_preference: option_title -> option_id
    ep = brief_body.get("emerging_preference")
    if ep and isinstance(ep, dict):
        title = ep.pop("option_title", "").lower()
        ep["option_id"] = option_title_to_id.get(title, "")

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
    if not workspace.primary_question:
        raise ValueError("Primary question not confirmed — call /question-confirmation first")

    extract_job = await session.get(Job, extract_job_id)
    if extract_job is None or extract_job.status != JobStatus.done:
        raise ValueError(f"extract_brief job {extract_job_id} is not done")

    merged = extract_job.output_payload.get("merged_draft", {})

    # Build user prompt from merged candidates
    candidates_json = json.dumps({
        k: merged.get(k, [])
        for k in ("options", "criteria", "constraints", "claims",
                  "positions", "objections", "blockers", "separate_issues", "actions")
    }, indent=2)

    user = (
        f"Confirmed decision question: {workspace.primary_question}\n"
        f"Scope: {workspace.scope or 'Not specified'}\n\n"
        f"Extracted candidates from the full discussion:\n{candidates_json}\n\n"
        "Synthesize a structured decision brief."
    )

    llm = LLMClient()
    brief_body = await llm.extract(_SYNTH_SYSTEM, user, _SYNTH_TOOL, timeout=120)

    # Server assigns all IDs (LLM must not generate them)
    brief_body = _assign_server_ids(brief_body)

    # Validate and strip any source_unit_ids that don't belong to this workspace
    ref_warnings = await _validate_source_refs(brief_body, workspace_id, session)

    # Add the confirmed question metadata
    brief_body["question"] = {
        "text": workspace.primary_question,
        "scope": workspace.scope,
    }

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
