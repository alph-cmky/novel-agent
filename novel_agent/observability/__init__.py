"""Observability integration — LangFuse tracing for LLM calls."""

from novel_agent.observability.langfuse import (
    create_trace,
    get_handler,
    is_configured,
    score_trace,
    set_handler,
)

__all__ = [
    "create_trace",
    "get_handler",
    "is_configured",
    "score_trace",
    "set_handler",
]
