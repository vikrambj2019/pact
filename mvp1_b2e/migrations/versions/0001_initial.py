"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table("workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("primary_question", sa.Text, nullable=True),
        sa.Column("scope", sa.Text, nullable=True),
        sa.Column("lifecycle_status", sa.String(50), nullable=False, server_default="draft"),
        sa.Column("current_revision_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decision_owner_label", sa.String(300), nullable=True),
        sa.Column("decision_rule", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table("events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("actor_type", sa.String(20), nullable=False),
        sa.Column("actor_label", sa.String(300), nullable=True),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("prior_revision", sa.Integer, nullable=True),
        sa.Column("resulting_revision", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_workspace_id", "events", ["workspace_id"])

    op.create_table("brief_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("revision_number", sa.Integer, nullable=False),
        sa.Column("parent_revision_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("brief_revisions.id"), nullable=True),
        sa.Column("structured_snapshot", postgresql.JSONB, nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("events.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_brief_revisions_workspace_id", "brief_revisions", ["workspace_id"])

    op.create_foreign_key("fk_workspaces_current_revision", "workspaces", "brief_revisions",
                          ["current_revision_id"], ["id"], use_alter=True)

    op.create_table("participants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("display_label", sa.String(300), nullable=False),
        sa.Column("source_aliases", postgresql.JSONB, nullable=False),
        sa.Column("resolution_status", sa.String(20), nullable=False, server_default="unresolved"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_participants_workspace_id", "participants", ["workspace_id"])

    op.create_table("source_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("original_filename", sa.String(500), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(200), nullable=False),
        sa.Column("parser_version", sa.String(20), nullable=False, server_default="1.0"),
        sa.Column("parse_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("warnings", postgresql.JSONB, nullable=False),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_source_documents_workspace_id", "source_documents", ["workspace_id"])

    op.create_table("source_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("source_documents.id"), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("sequence", sa.Integer, nullable=False),
        sa.Column("author_label", sa.String(300), nullable=True),
        sa.Column("mapped_participant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("participants.id"), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("original_text", sa.Text, nullable=False),
        sa.Column("normalized_text", sa.Text, nullable=False),
        sa.Column("locator", postgresql.JSONB, nullable=False),
        sa.Column("duplicate_of", postgresql.UUID(as_uuid=True), sa.ForeignKey("source_units.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_source_units_document_id", "source_units", ["document_id"])
    op.create_index("ix_source_units_workspace_id", "source_units", ["workspace_id"])

    op.create_table("conversation_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("author_label", sa.String(300), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("linked_revision_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("brief_revisions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversation_messages_workspace_id", "conversation_messages", ["workspace_id"])

    op.create_table("change_proposals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("base_revision_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("brief_revisions.id"), nullable=False),
        sa.Column("typed_operations", postgresql.JSONB, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("source_refs", postgresql.JSONB, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("rejection_rationale", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_change_proposals_workspace_id", "change_proposals", ["workspace_id"])

    op.create_table("decision_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("artifact_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("originating_revision_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("brief_revisions.id"), nullable=False),
        sa.Column("decision_owner_label", sa.String(300), nullable=False),
        sa.Column("recorded_by_label", sa.String(300), nullable=False),
        sa.Column("authority_verification", sa.String(50), nullable=False, server_default="not_verified_phase_1"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("immutable_snapshot", postgresql.JSONB, nullable=False),
        sa.Column("supersedes_artifact_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("decision_artifacts.id"), nullable=True),
        sa.Column("withdrawal_reason", sa.Text, nullable=True),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_decision_artifacts_workspace_id", "decision_artifacts", ["workspace_id"])

    op.create_table("jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("input_payload", postgresql.JSONB, nullable=False),
        sa.Column("output_payload", postgresql.JSONB, nullable=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("error_detail", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_workspace_id", "jobs", ["workspace_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_unique_constraint("uq_jobs_idempotency_key", "jobs", ["idempotency_key"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("decision_artifacts")
    op.drop_table("change_proposals")
    op.drop_table("conversation_messages")
    op.drop_table("source_units")
    op.drop_table("source_documents")
    op.drop_table("participants")
    op.drop_constraint("fk_workspaces_current_revision", "workspaces", type_="foreignkey")
    op.drop_table("brief_revisions")
    op.drop_table("events")
    op.drop_table("workspaces")
