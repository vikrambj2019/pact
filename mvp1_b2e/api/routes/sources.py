import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from ..db import get_db
from ..storage import get_storage, LocalStorage
from ..models import Workspace, SourceDocument, SourceUnit, Job, ParseStatus, JobType, JobStatus
from ..schemas.source import SourceUploadResponse, SourceUnitResponse
from ..config import settings

router = APIRouter(tags=["sources"])

ALLOWED_TYPES = {"text/plain", "text/markdown", "message/rfc822", "application/pdf"}
SUFFIX_MAP = {
    ".txt": "txt", ".md": "markdown", ".eml": "email_chain", ".pdf": "pdf",
}


@router.post("/workspaces/{workspace_id}/sources", response_model=SourceUploadResponse, status_code=202)
async def upload_source(
    workspace_id: uuid.UUID,
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not file and not pasted_text:
        raise HTTPException(422, "Provide a file upload or pasted_text")

    if file:
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(422, f"File exceeds {settings.max_upload_bytes} byte limit")
        filename = file.filename or "upload"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        doc_type = SUFFIX_MAP.get(suffix, "txt")
        storage_key, content_hash = await storage.put(data, suffix=suffix)
    else:
        data = pasted_text.encode()
        filename = None
        doc_type = "pasted_text"
        storage_key, content_hash = await storage.put(data, suffix=".txt")

    word_count_estimate = len(data.decode(errors="replace").split())

    ikey = idempotency_key or f"{workspace_id}:{content_hash}"
    doc = SourceDocument(
        workspace_id=workspace_id,
        type=doc_type,
        original_filename=filename,
        content_hash=content_hash,
        storage_key=storage_key,
        parse_status=ParseStatus.pending,
        word_count=word_count_estimate,
    )
    db.add(doc)
    await db.flush()

    job = Job(
        workspace_id=workspace_id,
        type=JobType.parse_source,
        status=JobStatus.pending,
        input_payload={"source_document_id": str(doc.id)},
        idempotency_key=ikey,
    )
    db.add(job)
    await db.commit()

    return SourceUploadResponse(
        source_document_id=doc.id,
        job_id=job.id,
        filename=filename,
        content_hash=content_hash,
        word_count_estimate=word_count_estimate,
    )


@router.get("/sources/{source_id}/units/{unit_id}", response_model=SourceUnitResponse)
async def get_source_unit(
    source_id: uuid.UUID,
    unit_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    unit = await db.get(SourceUnit, unit_id)
    if not unit or unit.document_id != source_id:
        raise HTTPException(404, "Source unit not found")
    return unit
