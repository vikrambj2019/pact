import uuid
import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, ForeignKey, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, UUIDMixin, TimestampMixin, utcnow


class LifecycleStatus(str, enum.Enum):
    draft = "draft"
    deliberating = "deliberating"
    decision_recorded = "decision_recorded"


class Workspace(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    primary_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scope: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lifecycle_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=LifecycleStatus.draft
    )
    current_revision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brief_revisions.id", use_alter=True), nullable=True
    )
    decision_owner_label: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    decision_rule: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    revisions: Mapped[list["BriefRevision"]] = relationship(
        "BriefRevision", back_populates="workspace",
        foreign_keys="BriefRevision.workspace_id", lazy="select"
    )
    events: Mapped[list["Event"]] = relationship("Event", back_populates="workspace", lazy="select")


class BriefRevision(UUIDMixin, Base):
    __tablename__ = "brief_revisions"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_revision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brief_revisions.id"), nullable=True
    )
    structured_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", use_alter=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(
        "Workspace", back_populates="revisions", foreign_keys=[workspace_id]
    )


class Event(UUIDMixin, Base):
    __tablename__ = "events"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(20), nullable=False)  # human / ai / system
    actor_label: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    prior_revision: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    resulting_revision: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="events")
