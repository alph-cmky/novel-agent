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
    APPROVED = "approved"


class OutlineStatus(StrEnum):
    """Lifecycle status of an outline entry (``outlines.status``).

    Shown in the chapter list UI.
    """

    PENDING = "pending"
    WRITING = "writing"
    DRAFTED = "drafted"
    APPROVED = "approved"
