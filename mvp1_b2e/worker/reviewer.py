"""Claim evidence review — scans all source units for supporting, contradicting, and limiting passages.

Flow:
  1. Load claim from current BriefRevision by claim_id.
  2. Load all non-duplicate SourceUnits for the workspace.
  3. Chunk units; for each chunk ask the LLM to find evidence passages.
  4. Validate every quote (must exist in original_text); compute char offsets.
  5. Aggregate verdict counts → new evidence_status.
  6. Write updated claim back to a new BriefRevision.
"""
from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.models import BriefRevision, Event, SourceUnit, Workspace


# ---------------------------------------------------------------------------
# LLM tool
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """\
You are a precise evidence analyst. You are given a claim and a set of discussion passages.
Your job is to identify passages that directly support, contradict, or limit the claim.

Rules:
- Only include passages with a clear, direct relationship to the claim. Skip tangential content.
- "supports" means the passage provides evidence the claim is true within this discussion.
- "contradicts" means the passage provides evidence the claim is false or contested.
- "limits" means the passage constrains the scope or applicability of the claim.
- The quote field must be an exact substring of the passage text — do not paraphrase or alter it.
- Do not invent evidence. If a passage is irrelevant, omit it entirely.
- If no passages are relevant, return an empty findings list.
"""

_REVIEW_TOOL = {
    "name": "evidence_findings",
    "description": "Passages that support, contradict, or limit the claim.",
    "input_schema": {
        "type": "object",
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source_unit_id", "quote", "verdict", "explanation"],
                    "properties": {
                        "source_unit_id": {"type": "string"},
                        "quote": {"type": "string",
                                  "description": "Exact verbatim substring of the passage text."},
                        "verdict": {"type": "string",
                                    "enum": ["supports", "contradicts", "limits"]},
                        "explanation": {"type": "string",
                                        "description": "One sentence explaining the relationship."}
                    }
                }
            }
        }
    }
}


def _format_units_for_review(units: list[dict]) -> str:
    lines = []
    for u in units:
        ts = f"[{u['locator'].get('timestamp_str', '')}] " if u["locator"].get("timestamp_str") else ""
        author = f"{u['author_label']}: " if u.get("author_label") else ""
        lines.append(f"[unit:{u['id']}]\n{ts}{author}{u['normalized_text']}")
    return "\n\n".join(lines)


async def _review_chunk(claim_text: str, units: list[dict], llm) -> list[dict]:
    user = (
        f"Claim to review:\n\"{claim_text}\"\n\n"
        f"Discussion passages:\n{_format_units_for_review(units)}"
    )
    result = await llm.extract(_REVIEW_SYSTEM, user, _REVIEW_TOOL, timeout=60)
    return result.get("findings", [])


# ---------------------------------------------------------------------------
# Quote validation
# ---------------------------------------------------------------------------

def _validate_findings(
    findings: list[dict],
    unit_map: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """Validate each finding's quote against the source unit's original_text.

    Returns (valid_findings, warnings).
    Each valid finding gets `start_char`, `end_char`, and `source_document_id` added.
    """
    valid: list[dict] = []
    warnings: list[str] = []

    for f in findings:
        unit_id = f.get("source_unit_id", "")
        unit = unit_map.get(unit_id)
        if unit is None:
            warnings.append(f"source_unit_id '{unit_id}' not found in workspace — skipped")
            continue
        quote = f.get("quote", "")
        original = unit["original_text"]
        pos = original.find(quote)
        if pos == -1:
            # Offsets must be into original_text (spec §5 source reference contract).
            # A quote that only exists in normalized text has no valid citation offset.
            warnings.append(
                f"Quote not found in original_text of unit {unit_id[:8]}: {quote[:60]!r} — skipped"
            )
            continue
        valid.append({
            "source_unit_id": unit_id,
            "source_document_id": unit["document_id"],
            "quote": quote,
            "start_char": pos,
            "end_char": pos + len(quote),
            "verdict": f["verdict"],
            "explanation": f.get("explanation", ""),
        })

    return valid, warnings


# ---------------------------------------------------------------------------
# Evidence status aggregation
# ---------------------------------------------------------------------------

def _aggregate_status(findings: list[dict]) -> str:
    if not findings:
        return "unresolved"
    verdicts = {f["verdict"] for f in findings}
    has_support = "supports" in verdicts
    has_contra = "contradicts" in verdicts
    if has_support and has_contra:
        return "conflicting"
    if has_contra:
        return "contradicted_within_scope"
    if has_support:
        return "supported_within_scope"
    return "unresolved"  # only "limits" findings


# ---------------------------------------------------------------------------
# Main job entry point
# ---------------------------------------------------------------------------

async def run_claim_review(payload: dict, session: AsyncSession) -> dict:
    from .chunker import chunk_units
    from .llm import LLMClient

    workspace_id = uuid.UUID(payload["workspace_id"])
    claim_id: str = payload["claim_id"]
    base_revision_id = uuid.UUID(payload["base_revision_id"])

    workspace = await session.get(Workspace, workspace_id)
    if workspace is None:
        raise ValueError(f"Workspace {workspace_id} not found")

    # Load the revision the review was requested against
    base_rev = await session.get(BriefRevision, base_revision_id)
    if base_rev is None:
        raise ValueError(f"BriefRevision {base_revision_id} not found")

    snapshot = base_rev.structured_snapshot
    claim = next((c for c in snapshot.get("claims", []) if c.get("id") == claim_id), None)
    if claim is None:
        raise ValueError(f"Claim '{claim_id}' not found in revision {base_revision_id}")

    # Load all non-duplicate source units
    result = await session.execute(
        select(SourceUnit)
        .where(
            SourceUnit.workspace_id == workspace_id,
            SourceUnit.duplicate_of.is_(None),
        )
        .order_by(SourceUnit.sequence)
    )
    db_units = result.scalars().all()
    if not db_units:
        return {
            "claim_id": claim_id,
            "evidence_status": "unresolved",
            "findings": [],
            "warnings": ["No source units found in workspace"],
            "units_searched": 0,
        }

    units_data = [
        {
            "id": str(u.id),
            "sequence": u.sequence,
            "author_label": u.author_label,
            "normalized_text": u.normalized_text,
            "original_text": u.original_text,
            "document_id": str(u.document_id),
            "locator": u.locator or {},
        }
        for u in db_units
    ]
    unit_map = {u["id"]: u for u in units_data}

    # Chunked review
    llm = LLMClient()
    chunks = chunk_units(units_data, max_words=1500)
    all_raw_findings: list[dict] = []
    for chunk in chunks:
        chunk_findings = await _review_chunk(claim["text"], chunk, llm)
        all_raw_findings.extend(chunk_findings)

    # Validate quotes and compute char offsets
    valid_findings, warnings = _validate_findings(all_raw_findings, unit_map)
    evidence_status = _aggregate_status(valid_findings)

    # Re-query workspace under FOR UPDATE after LLM calls — the pre-call object may be
    # stale if another request updated current_revision_id while the LLM was running.
    locked_ws = await session.execute(
        select(Workspace)
        .where(Workspace.id == workspace_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    workspace = locked_ws.scalar_one()
    current_rev_id = workspace.current_revision_id or base_revision_id
    current_rev = await session.get(BriefRevision, current_rev_id)
    current_snapshot = current_rev.structured_snapshot if current_rev else snapshot
    current_rev_num = current_rev.revision_number if current_rev else base_rev.revision_number

    new_snapshot = copy.deepcopy(current_snapshot)
    claim_found = False
    for c in new_snapshot.get("claims", []):
        if c.get("id") == claim_id:
            c["evidence_status"] = evidence_status
            c["evidence_links"] = valid_findings
            claim_found = True
            break

    if not claim_found:
        # Claim was removed from the brief while the review was running — nothing to save.
        warnings.append(f"Claim '{claim_id}' no longer exists in current revision — evidence not saved")
        return {
            "claim_id": claim_id,
            "claim_text": claim["text"],
            "evidence_status": evidence_status,
            "findings": valid_findings,
            "warnings": warnings,
            "units_searched": len(units_data),
            "revision_id": None,
            "revision_number": None,
        }

    event = Event(
        workspace_id=workspace_id,
        event_type="claim_reviewed",
        actor_type="ai",
        actor_label="pact-reviewer",
        payload={
            "claim_id": claim_id,
            "evidence_status": evidence_status,
            "findings_count": len(valid_findings),
            "warnings": warnings,
        },
        prior_revision=current_rev_num,
        resulting_revision=current_rev_num + 1,
    )
    session.add(event)
    await session.flush()

    new_rev = BriefRevision(
        workspace_id=workspace_id,
        revision_number=current_rev_num + 1,
        parent_revision_id=current_rev_id,
        structured_snapshot=new_snapshot,
        event_id=event.id,
    )
    session.add(new_rev)
    await session.flush()

    workspace.current_revision_id = new_rev.id
    await session.commit()

    return {
        "claim_id": claim_id,
        "claim_text": claim["text"],
        "evidence_status": evidence_status,
        "findings": valid_findings,
        "warnings": warnings,
        "units_searched": len(units_data),
        "revision_id": str(new_rev.id),
        "revision_number": new_rev.revision_number,
    }
