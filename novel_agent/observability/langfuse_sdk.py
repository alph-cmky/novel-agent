"""Langfuse v4 SDK adapter. The rest of the app must not import langfuse directly."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def keys_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def base_url() -> str:
    return (os.getenv("LANGFUSE_BASE_URL") or "").rstrip("/")


def sdk_available() -> bool:
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


def get_sdk():
    from langfuse import get_client

    url = base_url()
    if url:
        os.environ["LANGFUSE_BASE_URL"] = url
    return get_client()


def create_trace_id() -> str:
    return get_sdk().create_trace_id()


def callback_handler(*, trace_id: str):
    from langfuse.langchain import CallbackHandler

    return CallbackHandler(trace_context={"trace_id": trace_id})


def start_chapter_span(*, name: str, trace_id: str, input_payload: Mapping[str, Any]):
    return get_sdk().start_as_current_observation(
        as_type="span",
        name=name,
        trace_context={"trace_id": trace_id},
        input=dict(input_payload),
    )


def propagate(*, session_id: str, trace_name: str, tags: list[str], metadata: dict[str, Any]):
    from langfuse import propagate_attributes

    return propagate_attributes(
        session_id=session_id or None,
        trace_name=trace_name,
        tags=tags or None,
        metadata=metadata or None,
    )


def score_current(name: str, value: float) -> None:
    get_sdk().score_current_trace(name=name, value=value)


def create_event(*, name: str, metadata: Mapping[str, Any] | None = None) -> None:
    get_sdk().create_event(name=name, metadata=dict(metadata or {}))


def start_tool_span(*, name: str, tool_input: Mapping[str, Any]):
    return get_sdk().start_as_current_observation(
        as_type="tool",
        name=name,
        input=dict(tool_input),
    )


def flush() -> None:
    if not sdk_available() or not keys_configured():
        return
    get_sdk().flush()
