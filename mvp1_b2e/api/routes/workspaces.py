import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_db
from ..models import Workspace, BriefRevision, Event, LifecycleStatus
from ..schemas.workspace import WorkspaceCreate, WorkspaceResponse, QuestionConfirmation

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=201)
async def create_workspace(body: WorkspaceCreate, db: AsyncSession = Depends(get_db)):
    ws = Workspace(
        title=body.title,
        primary_question=body.decision_question,
        decision_rule=body.decision_rule,
        lifecycle_status=LifecycleStatus.draft,
    )
    db.add(ws)
    event = Event(
        workspace_id=ws.id, event_type="workspace_created",
        actor_type="human", payload={"title": body.title},
    )
    db.add(event)
    await db.commit()
    await db.refresh(ws)
    return ws


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(workspace_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    return ws


@router.post("/{workspace_id}/question-confirmation", response_model=WorkspaceResponse)
async def confirm_question(
    workspace_id: uuid.UUID,
    body: QuestionConfirmation,
    db: AsyncSession = Depends(get_db),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    current_rev = ws.current_revision_id
    current_num = 0
    if current_rev:
        rev = await db.get(BriefRevision, current_rev)
        current_num = rev.revision_number if rev else 0
    if current_num != body.expected_revision:
        raise HTTPException(409, f"Stale revision: expected {body.expected_revision}, current {current_num}")

    ws.primary_question = body.primary_question
    ws.scope = body.scope
    ws.decision_owner_label = body.decision_owner_label
    ws.decision_rule = body.decision_rule or ws.decision_rule
    ws.lifecycle_status = LifecycleStatus.deliberating

    event = Event(
        workspace_id=ws.id, event_type="question_confirmed",
        actor_type="human",
        payload={"primary_question": body.primary_question, "scope": body.scope},
        prior_revision=current_num,
    )
    db.add(event)
    await db.commit()
    await db.refresh(ws)
    return ws
