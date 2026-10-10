import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class SourceUploadResponse(BaseModel):
    source_document_id: uuid.UUID
    job_id: uuid.UUID
    filename: Optional[str]
    content_hash: str
    word_count_estimate: Optional[int]


class SourceUnitResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    sequence: int
    author_label: Optional[str]
    timestamp: Optional[datetime]
    original_text: str
    locator: dict

    model_config = {"from_attributes": True}
