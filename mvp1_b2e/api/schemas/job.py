import uuid
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel


class JobResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    type: str
    status: str
    attempt_count: int
    error_code: Optional[str]
    error_detail: Optional[str]
    output_payload: Optional[dict]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
