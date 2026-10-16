"""Participant upsert — creates or finds Participant records by author label."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.models import Participant


async def upsert_participants(
    workspace_id: uuid.UUID,
    author_labels: list[str],
    session: AsyncSession,
) -> dict[str, uuid.UUID]:
    """
    For each label, find an existing Participant (by display_label or source_aliases)
    or create a new one. Returns {author_label: participant_id}.
    """
    if not author_labels:
        return {}

    result = await session.execute(
        select(Participant).where(Participant.workspace_id == workspace_id)
    )
    existing = result.scalars().all()

    # Build alias index: normalized label -> participant
    alias_index: dict[str, Participant] = {}
    for p in existing:
        alias_index[p.display_label.lower()] = p
        for alias in (p.source_aliases or []):
            alias_index[alias.lower()] = p

    mapping: dict[str, uuid.UUID] = {}
    for label in author_labels:
        key = label.lower()
        if key in alias_index:
            mapping[label] = alias_index[key].id
        else:
            participant = Participant(
                workspace_id=workspace_id,
                display_label=label,
                source_aliases=[label],
            )
            session.add(participant)
            await session.flush()
            alias_index[key] = participant
            mapping[label] = participant.id

    return mapping
