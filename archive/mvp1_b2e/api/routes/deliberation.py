import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import (
    BriefRevision, ChangeProposal, ConversationMessage, Event, Workspace,
)
from ..schemas.deliberation import (
    AcceptProposal, MessageCreate, MessageResponse, ProposalResponse,
    ProposalSummary, ProposedOperation, RejectProposal,
)

router = APIRouter(tags=["deliberation"])

_HISTORY_LIMIT = 12  # messages of context sent to the LLM


# ---------------------------------------------------------------------------
# POST /workspaces/{id}/messages  — send a chat message
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/messages", response_model=MessageResponse)
async def send_message(
    workspace_id: uuid.UUID,
    body: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    if not ws.current_revision_id:
        raise HTTPException(422, "No brief available — run extraction and synthesis first")

    revision = await db.get(BriefRevision, ws.current_revision_id)
    if revision is None:
        raise HTTPException(404, "Current revision not found")

    if body.expected_revision is not None and body.expected_revision != revision.revision_number:
        raise HTTPException(
            409,
            f"Stale revision: expected {body.expected_revision}, current is {revision.revision_number}",
        )

    # Save user message
    user_msg = ConversationMessage(
        workspace_id=workspace_id,
        author_label=body.author_label,
        role="human",
        text=body.text,
        linked_revision_id=revision.id,
    )
    db.add(user_msg)
    await db.flush()

    # Load recent history (excluding the message we just saved)
    history_result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.workspace_id == workspace_id,
               ConversationMessage.id != user_msg.id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(_HISTORY_LIMIT)
    )
    history = [
        {"role": m.role, "author_label": m.author_label, "text": m.text}
        for m in reversed(history_result.scalars().all())
    ]

    # Run chat engine
    from ...worker.chat import run_chat
    from ...worker.llm import LLMClient

    llm = LLMClient()
    chat_result = await run_chat(
        snapshot=revision.structured_snapshot,
        revision_number=revision.revision_number,
        history=history,
        author_label=body.author_label,
        message_text=body.text,
        llm=llm,
    )

    # Save AI response
    ai_msg = ConversationMessage(
        workspace_id=workspace_id,
        author_label="pact-assistant",
        role="assistant",
        text=chat_result["response_text"],
        linked_revision_id=revision.id,
    )
    db.add(ai_msg)
    await db.flush()

    # If operations proposed, create a ChangeProposal
    proposal_summary: Optional[ProposalSummary] = None
    operations = chat_result.get("operations", [])
    if operations:
        proposal = ChangeProposal(
            workspace_id=workspace_id,
            base_revision_id=revision.id,
            typed_operations=operations,
            rationale=_build_rationale(operations, chat_result.get("response_text", "")),
            source_refs=[],
            status="pending",
        )
        db.add(proposal)
        await db.flush()

        proposal_summary = ProposalSummary(
            id=proposal.id,
            status=proposal.status,
            rationale=proposal.rationale,
            operations=[ProposedOperation(**op) for op in operations],
            created_at=proposal.created_at,
        )

    await db.commit()

    return MessageResponse(
        message_id=ai_msg.id,
        response_text=chat_result["response_text"],
        request_type=chat_result.get("request_type", "other"),
        follow_up_question=chat_result.get("follow_up_question"),
        proposal=proposal_summary,
    )


# ---------------------------------------------------------------------------
# GET /workspaces/{id}/messages  — conversation history
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/messages")
async def list_messages(
    workspace_id: uuid.UUID,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")

    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.workspace_id == workspace_id)
        .order_by(ConversationMessage.created_at.asc())
        .limit(min(limit, 200))
    )
    messages = result.scalars().all()
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "author_label": m.author_label,
            "text": m.text,
            "linked_revision_id": str(m.linked_revision_id) if m.linked_revision_id else None,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# POST /workspaces/{id}/proposals/{proposal_id}/accept
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/proposals/{proposal_id}/accept")
async def accept_proposal(
    workspace_id: uuid.UUID,
    proposal_id: uuid.UUID,
    body: AcceptProposal,
    db: AsyncSession = Depends(get_db),
):
    # Lock the workspace row for the duration of this transaction so concurrent
    # accepts on the same workspace serialize rather than racing.
    ws_result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id).with_for_update()
    )
    ws = ws_result.scalar_one_or_none()
    if not ws:
        raise HTTPException(404, "Workspace not found")

    proposal = await db.get(ChangeProposal, proposal_id)
    if not proposal or proposal.workspace_id != workspace_id:
        raise HTTPException(404, "Proposal not found")
    if proposal.status != "pending":
        raise HTTPException(409, f"Proposal is already '{proposal.status}'")

    # Re-read current revision under the lock so we see the latest state
    current_rev = await db.get(BriefRevision, ws.current_revision_id)
    if current_rev is None:
        raise HTTPException(404, "Current revision not found")
    if current_rev.revision_number != body.expected_revision:
        raise HTTPException(
            409,
            f"Stale revision: expected {body.expected_revision}, current is {current_rev.revision_number}",
        )
    if current_rev.id != proposal.base_revision_id:
        raise HTTPException(
            409,
            "Proposal was based on an older revision — reject it and send a new message for a fresh proposal",
        )

    # Apply operations — any skipped operation means the proposal is malformed;
    # refuse to create a revision that only partially reflects what was approved.
    from ...worker.operations import apply_operations

    new_snapshot, op_warnings = apply_operations(
        current_rev.structured_snapshot,
        proposal.typed_operations,
    )
    if op_warnings:
        raise HTTPException(
            422,
            {
                "error": "One or more operations could not be applied; proposal not accepted",
                "warnings": op_warnings,
                "hint": "Reject this proposal and send a new message to get a corrected one",
            },
        )

    # Create new revision
    event = Event(
        workspace_id=workspace_id,
        event_type="proposal_accepted",
        actor_type="human",
        actor_label="operator",
        payload={
            "proposal_id": str(proposal_id),
            "op_count": len(proposal.typed_operations),
        },
        prior_revision=current_rev.revision_number,
        resulting_revision=current_rev.revision_number + 1,
    )
    db.add(event)
    await db.flush()

    new_rev = BriefRevision(
        workspace_id=workspace_id,
        revision_number=current_rev.revision_number + 1,
        parent_revision_id=current_rev.id,
        structured_snapshot=new_snapshot,
        event_id=event.id,
    )
    db.add(new_rev)
    await db.flush()

    ws.current_revision_id = new_rev.id
    proposal.status = "accepted"
    proposal.resolved_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "proposal_id": str(proposal_id),
        "status": "accepted",
        "new_revision_number": new_rev.revision_number,
        "new_revision_id": str(new_rev.id),
        "warnings": op_warnings,
    }


# ---------------------------------------------------------------------------
# POST /workspaces/{id}/proposals/{proposal_id}/reject
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/proposals/{proposal_id}/reject")
async def reject_proposal(
    workspace_id: uuid.UUID,
    proposal_id: uuid.UUID,
    body: RejectProposal,
    db: AsyncSession = Depends(get_db),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")

    proposal = await db.get(ChangeProposal, proposal_id)
    if not proposal or proposal.workspace_id != workspace_id:
        raise HTTPException(404, "Proposal not found")
    if proposal.status != "pending":
        raise HTTPException(409, f"Proposal is already '{proposal.status}'")

    proposal.status = "rejected"
    proposal.rejection_rationale = body.rationale
    proposal.resolved_at = datetime.now(timezone.utc)
    await db.commit()

    return {"proposal_id": str(proposal_id), "status": "rejected"}


# ---------------------------------------------------------------------------
# GET /workspaces/{id}/proposals/{proposal_id}
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/proposals/{proposal_id}", response_model=ProposalResponse)
async def get_proposal(
    workspace_id: uuid.UUID,
    proposal_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    proposal = await db.get(ChangeProposal, proposal_id)
    if not proposal or proposal.workspace_id != workspace_id:
        raise HTTPException(404, "Proposal not found")
    return proposal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_rationale(operations: list[dict], response_text: str) -> str:
    types = ", ".join(sorted({op["type"] for op in operations}))
    summary = response_text[:200].rstrip()
    return f"[{types}] {summary}"
