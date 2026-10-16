import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import (
    BriefRevision, Job, JobStatus, JobType, ParseStatus, SourceDocument, Workspace,
)

router = APIRouter(tags=["brief"])


@router.post("/workspaces/{workspace_id}/brief/extract", status_code=202)
async def start_extraction(workspace_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Queue an extract_brief job. Returns immediately; poll /jobs/{id} for completion."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")

    result = await db.execute(
        select(SourceDocument).where(
            SourceDocument.workspace_id == workspace_id,
            SourceDocument.parse_status == ParseStatus.done,
        ).limit(1)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(422, "No parsed source documents found — upload and parse sources first")

    # Deduplicate only against already-running/pending jobs; allow re-runs after completion
    existing = await db.execute(
        select(Job).where(
            Job.workspace_id == workspace_id,
            Job.type == JobType.extract_brief,
            Job.status.in_([JobStatus.pending, JobStatus.running]),
        )
    )
    if ex := existing.scalar_one_or_none():
        return {"job_id": str(ex.id), "workspace_id": str(workspace_id), "status": ex.status}

    job = Job(
        workspace_id=workspace_id,
        type=JobType.extract_brief,
        status=JobStatus.pending,
        input_payload={"workspace_id": str(workspace_id)},
        idempotency_key=f"extract_brief:{workspace_id}:{uuid.uuid4()}",
    )
    db.add(job)
    await db.commit()
    return {"job_id": str(job.id), "workspace_id": str(workspace_id), "status": "pending"}


@router.post("/workspaces/{workspace_id}/brief/synthesize", status_code=202)
async def start_synthesis(workspace_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Queue a synthesize_brief job. Requires question-confirmation to have been called first."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")

    result = await db.execute(
        select(Job)
        .where(
            Job.workspace_id == workspace_id,
            Job.type == JobType.extract_brief,
            Job.status == JobStatus.done,
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    extract_job = result.scalar_one_or_none()
    if not extract_job:
        raise HTTPException(422, "No completed extraction job found — run /brief/extract first")

    existing = await db.execute(
        select(Job).where(
            Job.workspace_id == workspace_id,
            Job.type == JobType.synthesize_brief,
            Job.status.in_([JobStatus.pending, JobStatus.running]),
        )
    )
    if ex := existing.scalar_one_or_none():
        return {"job_id": str(ex.id), "workspace_id": str(workspace_id), "status": ex.status}

    job = Job(
        workspace_id=workspace_id,
        type=JobType.synthesize_brief,
        status=JobStatus.pending,
        input_payload={
            "workspace_id": str(workspace_id),
            "extract_job_id": str(extract_job.id),
        },
        idempotency_key=f"synthesize_brief:{workspace_id}:{uuid.uuid4()}",
    )
    db.add(job)
    await db.commit()
    return {"job_id": str(job.id), "workspace_id": str(workspace_id), "status": "pending"}


@router.get("/workspaces/{workspace_id}/brief")
async def get_brief(workspace_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return the current BriefRevision for a workspace."""
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(404, "No brief available yet — run extract and synthesize first")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if not revision:
        raise HTTPException(404, "Brief revision not found")

    return {
        "workspace_id": str(workspace_id),
        "revision_id": str(revision.id),
        "revision_number": revision.revision_number,
        "created_at": revision.created_at.isoformat(),
        "brief": revision.structured_snapshot,
    }
