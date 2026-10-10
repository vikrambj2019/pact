"""Decision recording routes — Step 6.

Endpoints:
  GET  /workspaces/{id}/decisions/preview          — preview artifact without recording
  POST /workspaces/{id}/decisions                  — record decision (atomic)
  GET  /workspaces/{id}/decisions                  — list all artifacts
  GET  /workspaces/{id}/decisions/{artifact_id}    — fetch single artifact
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import BriefRevision, DecisionArtifact, Event, Workspace

router = APIRouter(tags=["decisions"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_immutable_snapshot(
    workspace: Workspace,
    revision: BriefRevision,
    decision_owner_label: str,
    recorded_by_label: str,
    rationale: str,
    notes: str,
    recorded_at: datetime,
    selected_option_id: Optional[str] = None,
) -> dict:
    """Assemble the immutable snapshot that will be frozen into the artifact."""
    # Resolve the option title for human readability; None if not specified or not found
    selected_option_title: Optional[str] = None
    if selected_option_id:
        for opt in revision.structured_snapshot.get("options", []):
            if opt.get("id") == selected_option_id:
                selected_option_title = opt.get("title")
                break

    return {
        "workspace_title": workspace.title,
        "primary_question": workspace.primary_question,
        "scope": workspace.scope,
        "decision_rule": workspace.decision_rule,
        "revision_number": revision.revision_number,
        "revision_id": str(revision.id),
        "selected_option_id": selected_option_id,
        "selected_option_title": selected_option_title,
        "rationale": rationale,
        "brief": revision.structured_snapshot,
        "notes": notes,
        "decision_owner_label": decision_owner_label,
        "recorded_by_label": recorded_by_label,
        "recorded_at": recorded_at.isoformat(),
    }


def _artifact_summary(a: DecisionArtifact) -> dict:
    return {
        "artifact_id": str(a.id),
        "artifact_version": a.artifact_version,
        "decision_owner_label": a.decision_owner_label,
        "recorded_by_label": a.recorded_by_label,
        "recorded_at": a.recorded_at.isoformat(),
        "supersedes_artifact_id": str(a.supersedes_artifact_id) if a.supersedes_artifact_id else None,
        "withdrawn_at": a.withdrawn_at.isoformat() if a.withdrawn_at else None,
    }


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RecordDecisionRequest(BaseModel):
    decision_owner_label: str = Field(min_length=1)
    recorded_by_label: str = Field(min_length=1)
    expected_revision: int
    rationale: str = Field(min_length=1)
    selected_option_id: Optional[str] = None  # None means no specific option (e.g. defer)
    notes: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/decisions/preview")
async def preview_decision(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return what the decision artifact will look like without writing anything."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(422, "No brief available — synthesize first")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if revision is None:
        raise HTTPException(404, "Current revision not found")

    preview = _build_immutable_snapshot(
        ws, revision,
        decision_owner_label=ws.decision_owner_label or "",
        recorded_by_label="",
        rationale="",
        notes="",
        recorded_at=datetime.now(timezone.utc),
    )
    return {
        "workspace_id": str(workspace_id),
        "lifecycle_status": ws.lifecycle_status,
        "revision_number": revision.revision_number,
        "revision_id": str(revision.id),
        "preview": preview,
    }


@router.post("/workspaces/{workspace_id}/decisions", status_code=201)
async def record_decision(
    workspace_id: uuid.UUID,
    body: RecordDecisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Atomically record a decision: create immutable artifact, emit event, advance lifecycle."""
    # Lock workspace to prevent concurrent decision recording
    locked = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id).with_for_update()
    )
    ws = locked.scalar_one_or_none()
    if ws is None:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(422, "No brief available — synthesize first")
    if ws.lifecycle_status == "draft":
        raise HTTPException(422, "Workspace is still in draft — confirm the question first")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if revision is None:
        raise HTTPException(404, "Current revision not found")
    if revision.revision_number != body.expected_revision:
        raise HTTPException(
            409,
            f"Stale revision: expected {body.expected_revision}, "
            f"current {revision.revision_number}",
        )

    # Validate selected_option_id if provided
    if body.selected_option_id is not None:
        option_ids = {opt.get("id") for opt in revision.structured_snapshot.get("options", [])}
        if body.selected_option_id not in option_ids:
            raise HTTPException(
                422,
                f"selected_option_id '{body.selected_option_id}' not found in current brief options",
            )

    recorded_at = datetime.now(timezone.utc)
    snapshot = _build_immutable_snapshot(
        ws, revision,
        decision_owner_label=body.decision_owner_label,
        recorded_by_label=body.recorded_by_label,
        rationale=body.rationale,
        notes=body.notes,
        recorded_at=recorded_at,
        selected_option_id=body.selected_option_id,
    )

    # Find the most recent non-withdrawn artifact to set supersedes_artifact_id
    prev_result = await db.execute(
        select(DecisionArtifact)
        .where(
            DecisionArtifact.workspace_id == workspace_id,
            DecisionArtifact.withdrawn_at.is_(None),
        )
        .order_by(DecisionArtifact.created_at.desc())
        .limit(1)
    )
    prev_artifact = prev_result.scalar_one_or_none()
    next_version = (prev_artifact.artifact_version + 1) if prev_artifact else 1

    artifact = DecisionArtifact(
        workspace_id=workspace_id,
        artifact_version=next_version,
        originating_revision_id=revision.id,
        decision_owner_label=body.decision_owner_label,
        recorded_by_label=body.recorded_by_label,
        recorded_at=recorded_at,
        immutable_snapshot=snapshot,
        supersedes_artifact_id=prev_artifact.id if prev_artifact else None,
    )
    db.add(artifact)
    await db.flush()  # assign artifact.id before event payload

    event = Event(
        workspace_id=workspace_id,
        event_type="decision_recorded",
        actor_type="human",
        actor_label=body.recorded_by_label,
        payload={
            "artifact_id": str(artifact.id),
            "artifact_version": artifact.artifact_version,
            "decision_owner_label": body.decision_owner_label,
            "revision_number": revision.revision_number,
        },
        prior_revision=revision.revision_number,
        resulting_revision=revision.revision_number,
    )
    db.add(event)

    ws.lifecycle_status = "decision_recorded"
    await db.commit()
    await db.refresh(artifact)

    return {
        "artifact_id": str(artifact.id),
        "artifact_version": artifact.artifact_version,
        "workspace_id": str(workspace_id),
        "lifecycle_status": ws.lifecycle_status,
        "decision_owner_label": artifact.decision_owner_label,
        "recorded_by_label": artifact.recorded_by_label,
        "authority_verification": artifact.authority_verification,
        "recorded_at": artifact.recorded_at.isoformat(),
        "originating_revision_id": str(artifact.originating_revision_id),
        "supersedes_artifact_id": (
            str(artifact.supersedes_artifact_id) if artifact.supersedes_artifact_id else None
        ),
        "immutable_snapshot": artifact.immutable_snapshot,
    }


@router.get("/workspaces/{workspace_id}/decisions")
async def list_decisions(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")

    result = await db.execute(
        select(DecisionArtifact)
        .where(DecisionArtifact.workspace_id == workspace_id)
        .order_by(DecisionArtifact.created_at.desc())
    )
    artifacts = result.scalars().all()
    return {
        "workspace_id": str(workspace_id),
        "lifecycle_status": ws.lifecycle_status,
        "decisions": [_artifact_summary(a) for a in artifacts],
    }


@router.get("/workspaces/{workspace_id}/decisions/{artifact_id}")
async def get_decision(
    workspace_id: uuid.UUID,
    artifact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    artifact = await db.get(DecisionArtifact, artifact_id)
    if artifact is None or artifact.workspace_id != workspace_id:
        raise HTTPException(404, "Decision artifact not found")

    return {
        **_artifact_summary(artifact),
        "workspace_id": str(artifact.workspace_id),
        "authority_verification": artifact.authority_verification,
        "originating_revision_id": str(artifact.originating_revision_id),
        "withdrawal_reason": artifact.withdrawal_reason,
        "immutable_snapshot": artifact.immutable_snapshot,
    }
