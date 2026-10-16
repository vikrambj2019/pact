import uuid
from datetime import datetime
from pydantic import BaseModel


class SourceRef(BaseModel):
    source_document_id: uuid.UUID
    source_unit_id: uuid.UUID
    start_char: int
    end_char: int
    basis: str = "source_reported"


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
