import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import BriefRevision, Job, JobStatus, JobType, Workspace

router = APIRouter(tags=["claims"])


@router.post("/workspaces/{workspace_id}/claims/{claim_id}/review", status_code=202)
async def request_claim_review(
    workspace_id: uuid.UUID,
    claim_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Queue a claim_review job. Poll /jobs/{id} for results."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(422, "No brief available — synthesize first")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if revision is None:
        raise HTTPException(404, "Current revision not found")

    claim = next(
        (c for c in revision.structured_snapshot.get("claims", []) if c.get("id") == claim_id),
        None,
    )
    if claim is None:
        raise HTTPException(404, f"Claim '{claim_id}' not found in current brief")

    # Idempotency: reuse a running/pending job for the same claim+revision
    existing = await db.execute(
        select(Job).where(
            Job.workspace_id == workspace_id,
            Job.type == JobType.claim_review,
            Job.status.in_([JobStatus.pending, JobStatus.running]),
            Job.input_payload["claim_id"].astext == claim_id,
        )
    )
    if ex := existing.scalar_one_or_none():
        return {
            "job_id": str(ex.id),
            "claim_id": claim_id,
            "claim_text": claim.get("text", ""),
            "status": ex.status,
        }

    job = Job(
        workspace_id=workspace_id,
        type=JobType.claim_review,
        status=JobStatus.pending,
        input_payload={
            "workspace_id": str(workspace_id),
            "claim_id": claim_id,
            "base_revision_id": str(revision.id),
        },
        idempotency_key=f"claim_review:{workspace_id}:{claim_id}:{uuid.uuid4()}",
    )
    db.add(job)
    await db.commit()

    return {
        "job_id": str(job.id),
        "claim_id": claim_id,
        "claim_text": claim.get("text", ""),
        "status": "pending",
    }


@router.get("/workspaces/{workspace_id}/claims/{claim_id}")
async def get_claim(
    workspace_id: uuid.UUID,
    claim_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return a single claim with its current evidence_links from the current brief revision."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(422, "No brief available")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if revision is None:
        raise HTTPException(404, "Current revision not found")

    claim = next(
        (c for c in revision.structured_snapshot.get("claims", []) if c.get("id") == claim_id),
        None,
    )
    if claim is None:
        raise HTTPException(404, f"Claim '{claim_id}' not found in current brief")

    # Find the most recent completed review job for this claim
    review_result = await db.execute(
        select(Job)
        .where(
            Job.workspace_id == workspace_id,
            Job.type == JobType.claim_review,
            Job.status == JobStatus.done,
        )
        .order_by(Job.created_at.desc())
        .limit(10)
    )
    review_jobs = review_result.scalars().all()
    last_review = next(
        (j for j in review_jobs if j.input_payload.get("claim_id") == claim_id),
        None,
    )

    return {
        "workspace_id": str(workspace_id),
        "revision_id": str(revision.id),
        "revision_number": revision.revision_number,
        "claim": claim,
        "last_review": {
            "job_id": str(last_review.id),
            "units_searched": last_review.output_payload.get("units_searched"),
            "warnings": last_review.output_payload.get("warnings", []),
            "completed_at": last_review.updated_at.isoformat(),
        } if last_review else None,
    }
