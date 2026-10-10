import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    decision_question: Optional[str] = None
    decision_rule: Optional[str] = None


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    title: str
    primary_question: Optional[str]
    scope: Optional[str]
    lifecycle_status: str
    current_revision_id: Optional[uuid.UUID]
    decision_owner_label: Optional[str]
    decision_rule: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QuestionConfirmation(BaseModel):
    primary_question: str = Field(min_length=1)
    scope: Optional[str] = None
    decision_owner_label: Optional[str] = None
    decision_rule: Optional[str] = None
    expected_revision: int
