"""Chapter-level tracing: the only public observability API.

``chapter_trace`` / ``flush_tracing`` / ``require_tracing_config``.
Unconfigured or ``LANGFUSE_TRACING=0`` yields a NullHandle.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from novel_agent.config import env_bool
from novel_agent.observability import langfuse_sdk as sdk
from novel_agent.observability import payload as pl

logger = logging.getLogger(__name__)

_active_handle: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "chapter_trace_handle", default=None
)


def tracing_requested() -> bool:
    return env_bool("LANGFUSE_TRACING", False)


def capture_prompts() -> bool:
    return env_bool("LANGFUSE_CAPTURE_PROMPTS", False)


def require_tracing_config(*, strict: bool) -> str | None:
    """Return an error message when tracing cannot start.

    API uses ``strict=True`` (raises). Eval uses ``strict=False``
    (caller logs and continues with NullHandle).
    """
    if not tracing_requested():
        return None
    if not sdk.sdk_available():
        msg = "LANGFUSE_TRACING=1 需要 `uv sync --extra observability`"
    elif not sdk.keys_configured() or not sdk.base_url():
        msg = "LANGFUSE_TRACING=1 需要 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY、LANGFUSE_BASE_URL"
    else:
        msg = None
    if msg and strict:
        raise RuntimeError(msg)
    return msg


class NullHandle:
    enabled = False
    handler = None

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id

    def record_outcome(self, values: Mapping[str, Any], *, interrupted: bool = False) -> None:
        return

    def event(self, name: str, metadata: Mapping[str, Any] | None = None) -> None:
        return

    def tool_span(self, name: str, tool_input: Mapping[str, Any]):
        return _NullCM()


class _NullCM:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def update(self, **kwargs) -> None:
        return None


@dataclass
class TraceHandle:
    enabled: bool
    trace_id: str
    handler: Any = None
    _root: Any = None

    def record_outcome(self, values: Mapping[str, Any], *, interrupted: bool = False) -> None:
        try:
            output = pl.outcome_output(values, interrupted=interrupted)
            if self._root is not None:
                self._root.update(output=output)
            for score in pl.outcome_scores(values):
                sdk.score_current(score["name"], float(score["value"]))
            for event in pl.outcome_events(values, interrupted=interrupted):
                sdk.create_event(
                    name=event["name"],
                    metadata=event.get("metadata") or {},
                )
        except Exception:
            logger.exception("langfuse record_outcome failed")

    def event(self, name: str, metadata: Mapping[str, Any] | None = None) -> None:
        try:
            sdk.create_event(name=name, metadata=metadata)
        except Exception:
            logger.exception("langfuse event failed")

    def tool_span(self, name: str, tool_input: Mapping[str, Any]):
        try:
            return sdk.start_tool_span(name=name, tool_input=pl.redact_tool_args(tool_input))
        except Exception:
            logger.exception("langfuse tool span failed")
            return _NullCM()


def current_handle() -> NullHandle | TraceHandle:
    return _active_handle.get() or NullHandle()


def flush_tracing() -> None:
    try:
        sdk.flush()
    except Exception:
        logger.exception("langfuse flush failed")


@asynccontextmanager
async def chapter_trace(state: Mapping[str, Any], *, source: str = "api") -> AsyncIterator[Any]:
    """One chapter, one trace. Never raises into the writing path."""
    payload = dict(state)
    trace_id = str(payload.get("trace_id") or "")
    handle: NullHandle | TraceHandle
    if not tracing_requested() or require_tracing_config(strict=False) is not None:
        if tracing_requested():
            logger.warning("Langfuse tracing requested but not configured; using NullHandle")
        handle = NullHandle(trace_id or None)
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            _active_handle.reset(token)
        return

    try:
        if not trace_id:
            trace_id = sdk.create_trace_id()
        chapter_number = payload.get("chapter_number") or 0
        name = f"chapter.{chapter_number}"
        handler = sdk.callback_handler(trace_id=trace_id) if capture_prompts() else None
        span_cm = sdk.start_chapter_span(
            name=name, trace_id=trace_id, input_payload=pl.chapter_input(payload)
        )
        attr_cm = sdk.propagate(
            session_id=str(payload.get("project_id") or ""),
            trace_name=name,
            tags=pl.chapter_tags(payload, source=source),
            metadata=pl.compact_meta(payload),
        )
    except Exception:
        logger.exception("langfuse chapter_trace failed; continuing without tracing")
        handle = NullHandle(trace_id or None)
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            _active_handle.reset(token)
        return

    with span_cm as root, attr_cm:
        handle = TraceHandle(enabled=True, trace_id=trace_id, handler=handler, _root=root)
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            _active_handle.reset(token)
