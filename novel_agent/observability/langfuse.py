"""LangFuse integration via LangChain CallbackHandler.

Uses contextvars to propagate the handler across async tasks within a
single request, so every LLM call in the graph pipeline is traced.

Setup:
  Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally
  LANGFUSE_HOST (defaults to https://cloud.langfuse.com).

  If not configured, all functions are no-ops — no tracing overhead.
"""

import contextvars
import os

_langfuse_handler: contextvars.ContextVar = contextvars.ContextVar("langfuse_handler", default=None)
_trace: contextvars.ContextVar = contextvars.ContextVar("langfuse_trace", default=None)


def is_configured() -> bool:
    """Check if LangFuse credentials are set."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _get_client():
    """Lazy-init the LangFuse client (v3)."""
    from langfuse import get_client as _get

    return _get()


def set_handler(handler) -> None:
    """Set the LangFuse CallbackHandler for the current request context."""
    _langfuse_handler.set(handler)


def get_handler():
    """Get the LangFuse CallbackHandler for the current request context.

    Returns None if LangFuse is not configured or no handler was set.
    """
    if not is_configured():
        return None
    return _langfuse_handler.get()


def create_trace(
    name: str,
    project_id: str = "",
    chapter_number: int = 0,
    metadata: dict | None = None,
):
    """Create a LangFuse trace and set its handler in the current context.

    Returns the trace object (or None if not configured). The handler is
    automatically stored in the contextvar so all downstream LLM calls
    are grouped under this trace.

    Usage in API / SSE handler:
        create_trace("write_chapter_1", project_id="abc", chapter_number=1)
        # Run graph — all agent LLM calls auto-traced under this trace
    """
    if not is_configured():
        return None

    langfuse = _get_client()
    meta = metadata or {}
    if project_id:
        meta["project_id"] = project_id
    if chapter_number:
        meta["chapter_number"] = chapter_number

    trace = langfuse.trace(name=name, metadata=meta)
    _trace.set(trace)
    handler = trace.get_langchain_handler()
    _langfuse_handler.set(handler)
    return trace


def score_trace(scores: dict[str, float]) -> None:
    """Push scores to the current LangFuse trace.

    Safe to call at any point — does nothing if LangFuse is not
    configured or no trace is active in the current context.

    Usage after graph completion:
        score_trace({"editor_score": 82, "continuity_score": 90})
    """
    if not is_configured():
        return
    trace = _trace.get()
    if trace is None:
        return
    for name, value in scores.items():
        trace.score(name=name, value=value)
