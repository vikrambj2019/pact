"""Shared data types for the Pact decision-engine front end.

Pipeline:  transcript -> normalize() -> list[Span] -> classify() -> list[ClassifiedSpan]
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Input ──────────────────────────────────────────────────────────────────

class RawMessage(BaseModel):
    """One message exactly as the channel delivered it (Telegram msg, email, ...)."""
    id: str
    author: str                      # from channel metadata — never inferred
    ts: datetime
    text: str
    reply_to: str | None = None      # channel reply pointer, if any


class Transcript(BaseModel):
    channel: str = "telegram"
    messages: list[RawMessage]


# ── Normalizer output ──────────────────────────────────────────────────────

class Burst(BaseModel):
    """Consecutive messages by one author with no other speaker in between
    and no gap longer than the coalesce threshold. Built deterministically."""
    id: str
    author: str
    start_ts: datetime
    end_ts: datetime
    message_ids: list[str]
    texts: list[str]


class SpanStatus(str, Enum):
    active = "active"
    self_corrected = "self_corrected"   # overridden within the same burst (kept for audit)


class Span(BaseModel):
    """Smallest piece of text carrying at most one extractable element."""
    id: str
    burst_id: str
    author: str
    ts: datetime                         # ts of the message the span came from
    text: str                            # verbatim substring of a source message
    source_message_ids: list[str]        # usually 1; >1 only if a thought spans messages
    status: SpanStatus = SpanStatus.active
    superseded_by: str | None = None     # span id that corrected this one
    context_note: str | None = None      # e.g. "'the 5th' = proposed date for the hike"
    verified: bool = True                # False if text isn't a verbatim substring
    recovered: bool = False              # added by the coverage guard, not the model


# ── Classifier output ──────────────────────────────────────────────────────

class ElementType(str, Enum):
    candidate_question = "candidate_question"
    option = "option"
    criterion = "criterion"
    constraint = "constraint"
    claim = "claim"
    position = "position"
    agreement = "agreement"
    objection = "objection"
    blocker = "blocker"
    action = "action"
    separate_issue = "separate_issue"
    decision = "decision"        # the group closes a question / picks an option
    process = "process"          # facilitation: agenda, calls for input, objection checks
    irrelevant = "irrelevant"


# Labels that carry no decision content; dropped whenever a content label is present.
NON_CONTENT = {ElementType.process, ElementType.irrelevant}


class Label(BaseModel):
    type: ElementType
    confidence: float = Field(ge=0, le=1)
    gist: str = ""                  # one-line normalized reading, e.g. "proposes 5th as hike date"


class ClassifiedSpan(BaseModel):
    span: Span
    labels: list[Label]
    rationale: str = ""

    @property
    def is_irrelevant(self) -> bool:
        return all(l.type in NON_CONTENT for l in self.labels)
