import uuid
import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, ForeignKey, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, UUIDMixin, utcnow


class ParseStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class ResolutionStatus(str, enum.Enum):
    unresolved = "unresolved"
    resolved = "resolved"
    merged = "merged"


class SourceDocument(UUIDMixin, Base):
    __tablename__ = "source_documents"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # transcript, email_chain, pasted_text, txt, markdown, pdf
    original_filename: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256
    storage_key: Mapped[str] = mapped_column(String(200), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    parse_status: Mapped[str] = mapped_column(String(20), nullable=False, default=ParseStatus.pending)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    units: Mapped[list["SourceUnit"]] = relationship("SourceUnit", back_populates="document", lazy="select")


class SourceUnit(UUIDMixin, Base):
    __tablename__ = "source_units"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_documents.id"), nullable=False, index=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    author_label: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    mapped_participant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("participants.id"), nullable=True
    )
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    locator: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    duplicate_of: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_units.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    document: Mapped["SourceDocument"] = relationship("SourceDocument", back_populates="units")


class Participant(UUIDMixin, Base):
    __tablename__ = "participants"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True
    )
    display_label: Mapped[str] = mapped_column(String(300), nullable=False)
    source_aliases: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    resolution_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ResolutionStatus.unresolved
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
