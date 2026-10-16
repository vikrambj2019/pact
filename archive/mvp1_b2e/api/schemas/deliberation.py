import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class MessageCreate(BaseModel):
    author_label: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    expected_revision: Optional[int] = None


class ProposedOperation(BaseModel):
    type: str
    entity_id: Optional[str] = None
    fields: dict[str, Any] = {}
    rationale: Optional[str] = None


class ProposalSummary(BaseModel):
    id: uuid.UUID
    status: str
    rationale: str
    operations: list[ProposedOperation]
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message_id: uuid.UUID
    response_text: str
    request_type: str
    follow_up_question: Optional[str] = None
    proposal: Optional[ProposalSummary] = None


class ProposalResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    base_revision_id: uuid.UUID
    status: str
    rationale: str
    operations: list[dict]
    rejection_rationale: Optional[str] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AcceptProposal(BaseModel):
    expected_revision: int


class RejectProposal(BaseModel):
    rationale: Optional[str] = None
