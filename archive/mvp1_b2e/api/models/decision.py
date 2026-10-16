import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, ForeignKey, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, UUIDMixin, utcnow


class DecisionArtifact(UUIDMixin, Base):
    __tablename__ = "decision_artifacts"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    artifact_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    originating_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brief_revisions.id"), nullable=False
    )
    decision_owner_label: Mapped[str] = mapped_column(String(300), nullable=False)
    recorded_by_label: Mapped[str] = mapped_column(String(300), nullable=False)
    # Always "not_verified_phase_1" in Phase 1
    authority_verification: Mapped[str] = mapped_column(
        String(50), nullable=False, default="not_verified_phase_1"
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    immutable_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    supersedes_artifact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_artifacts.id"), nullable=True
    )
    withdrawal_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    withdrawn_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
