"""Shared status enums — single source of truth for chapter/outline states.

These are the wire-contract values persisted to SQLite and returned by the API.
The frontend mirrors them in ``frontend/src/lib/status.ts`` (which also holds the
display labels — a presentation concern that stays on the frontend).
"""

from enum import StrEnum


class ChapterStatus(StrEnum):
    """Status of a written chapter's content (``chapters.status``).

    Whether the chapter text has been human-approved.
    """

    DRAFT = "draft"
    WRITING = "writing"
    FAILED = "failed"
    APPROVED = "approved"


class OutlineStatus(StrEnum):
    """Lifecycle status of an outline entry (``outlines.status``).

    Shown in the chapter list UI.
    """

    PENDING = "pending"
    WRITING = "writing"
    FAILED = "failed"
    DRAFTED = "drafted"
    APPROVED = "approved"


class RunStatus(StrEnum):
    """Durable lifecycle of a generation/review run."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    WAITING_USER = "waiting_user"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ProposalStatus(StrEnum):
    """Review lifecycle of an Agent-proposed Canon change."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMMITTED = "committed"


class OutboxStatus(StrEnum):
    """Delivery lifecycle for derived-state events."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    STALE = "stale"


class EvolutionAction(StrEnum):
    """Control-flow action returned by deterministic evolution evaluation."""

    CONTINUE = "continue"
    STOP = "stop"
