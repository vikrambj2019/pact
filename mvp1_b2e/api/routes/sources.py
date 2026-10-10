import hashlib
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_db
from ..storage import get_storage, LocalStorage
from ..models import Workspace, SourceDocument, SourceUnit, Job, ParseStatus, JobType, JobStatus
from ..schemas.source import SourceUploadResponse, SourceUnitResponse
from ..config import settings

router = APIRouter(tags=["sources"])
SUFFIX_MAP = {".txt": "txt", ".md": "markdown", ".eml": "email_chain", ".pdf": "pdf"}


def upload_response(doc, job):
    return SourceUploadResponse(source_document_id=doc.id, job_id=job.id,
                                filename=doc.original_filename, content_hash=doc.content_hash,
                                word_count_estimate=doc.word_count)


@router.post("/workspaces/{workspace_id}/sources", response_model=SourceUploadResponse, status_code=202)
async def upload_source(
    workspace_id: uuid.UUID,
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Form(None),
    idempotency_header: Optional[str] = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
):
    # Serialize uploads per workspace: duplicate keys and cumulative limits are checked
    # in the same transaction as the source/job insertion.
    ws = (await db.execute(select(Workspace).where(Workspace.id == workspace_id)
                           .with_for_update())).scalar_one_or_none()
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if (file is not None) == (pasted_text is not None):
        raise HTTPException(422, "Provide exactly one file or pasted_text")
    if idempotency_header and idempotency_key and idempotency_header != idempotency_key:
        raise HTTPException(422, "Conflicting idempotency keys")
    if file is not None:
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix not in SUFFIX_MAP:
            raise HTTPException(422, "Supported extensions: .txt, .md, .eml, .pdf")
        data = await file.read(settings.max_upload_bytes + 1)
        doc_type = SUFFIX_MAP[suffix]
    else:
        filename, suffix, doc_type = None, ".txt", "pasted_text"
        data = pasted_text.encode("utf-8")
    if not data or not data.strip():
        raise HTTPException(422, "Source must not be empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(422, f"Source exceeds {settings.max_upload_bytes} byte limit")
    if doc_type == "pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(422, "Invalid PDF header")
        # The parser must enforce the extracted-word budget for PDFs.
        word_count = None
    else:
        try:
            text = data.decode("utf-8") if doc_type != "email_chain" else data.decode(errors="replace")
        except UnicodeDecodeError:
            raise HTTPException(422, "Text sources must use UTF-8")
        word_count = len(text.split())
    content_hash = hashlib.sha256(data).hexdigest()
    key = idempotency_header or idempotency_key or content_hash
    if not key.strip() or len(key) > 200:
        raise HTTPException(422, "Idempotency key must contain 1–200 characters")
    # Namespace keys by workspace and endpoint without exceeding the DB column size.
    ikey = f"source:{workspace_id}:{hashlib.sha256(key.encode()).hexdigest()}"
    existing = (await db.execute(select(Job).where(Job.idempotency_key == ikey))).scalar_one_or_none()
    if existing:
        doc = await db.get(SourceDocument, uuid.UUID(existing.input_payload["source_document_id"]))
        if not doc or (doc.content_hash, doc.type, doc.original_filename) != (content_hash, doc_type, filename):
            raise HTTPException(409, "Idempotency key already used for different input")
        return upload_response(doc, existing)
    total = (await db.execute(select(func.coalesce(func.sum(SourceDocument.word_count), 0))
                              .where(SourceDocument.workspace_id == workspace_id))).scalar_one()
    if word_count is not None and total + word_count > settings.max_workspace_words:
        raise HTTPException(422, "Workspace exceeds extracted-word limit")
    storage_key, _ = await storage.put(data, suffix=suffix)
    try:
        doc = SourceDocument(workspace_id=workspace_id, type=doc_type, original_filename=filename,
                             content_hash=content_hash, storage_key=storage_key,
                             parse_status=ParseStatus.pending, word_count=word_count)
        db.add(doc)
        await db.flush()
        job = Job(workspace_id=workspace_id, type=JobType.parse_source, status=JobStatus.pending,
                  input_payload={"source_document_id": str(doc.id)}, idempotency_key=ikey)
        db.add(job)
        await db.commit()
    except Exception:
        await db.rollback()
        await storage.delete(storage_key)
        raise
    return upload_response(doc, job)


@router.get("/sources/{source_id}/units/{unit_id}", response_model=SourceUnitResponse)
async def get_source_unit(source_id: uuid.UUID, unit_id: uuid.UUID,
                          db: AsyncSession = Depends(get_db)):
    unit = await db.get(SourceUnit, unit_id)
    if not unit or unit.document_id != source_id:
        raise HTTPException(404, "Source unit not found")
    return unit
