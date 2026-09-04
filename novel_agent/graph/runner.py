"""Single chapter-execution entry for API, CLI, and eval.

SSE and the eval adapter must not call ``graph.astream_events`` themselves —
otherwise interrupt handling, ``trace_id``, and later tracing diverge.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from novel_agent.observability.tracing import chapter_trace

GRAPH_INTERRUPT_EVENT = "novel_agent.graph_interrupt"
EVAL_AUTO_APPROVE = {"action": "approve", "comments": ""}

GraphEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class ChapterOutcome:
    """Result of one graph invocation (or a full auto-approve loop)."""

    values: dict[str, Any]
    interrupted: bool
    next_nodes: tuple[str, ...] = ()
    interrupt_payload: dict[str, Any] | None = None
    trace_id: str | None = None


@dataclass
class ChapterRunContext:
    payload: Any
    config: dict
    handle: Any
    trace_id: str | None


def chapter_payload(
    *,
    state: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any] | Command:
    """Build the LangGraph input: a state dict (with ``trace_id``) or a resume Command."""
    if resume is not None:
        return Command(resume=dict(resume))
    if state is None:
        raise ValueError("run_chapter requires state or resume")
    payload = dict(state)
    if not payload.get("trace_id"):
        payload["trace_id"] = uuid.uuid4().hex
    return payload


def _with_handler(config: dict, handler: Any) -> dict:
    merged = dict(config or {})
    if "configurable" in merged and isinstance(merged["configurable"], dict):
        merged["configurable"] = dict(merged["configurable"])
    if handler is None:
        return merged
    callbacks = list(merged.get("callbacks") or [])
    callbacks.append(handler)
    merged["callbacks"] = callbacks
    return merged


@asynccontextmanager
async def chapter_run_context(
    graph: Any,
    *,
    config: dict,
    state: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
    source: str = "api",
) -> AsyncIterator[ChapterRunContext]:
    """Bind ``chapter_trace`` + callbacks for one graph invocation (start or resume)."""
    payload = chapter_payload(state=state, resume=resume)
    trace_state: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
    if resume is not None:
        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values) if snapshot and snapshot.values else {}
        if values:
            trace_state = {**values, **trace_state}
    async with chapter_trace(trace_state, source=source) as handle:
        if isinstance(payload, dict) and handle.trace_id:
            payload["trace_id"] = handle.trace_id
        bound = _with_handler(config, handle.handler)
        yield ChapterRunContext(
            payload=payload,
            config=bound,
            handle=handle,
            trace_id=handle.trace_id,
        )


async def iterate_chapter_events(graph: Any, payload: Any, config: dict) -> Any:
    """Yield LangGraph ``astream_events`` items; surface interrupt as a sentinel event."""
    try:
        async for event in graph.astream_events(payload, config, version="v2"):
            yield event
    except GraphInterrupt as gi:
        data = gi.args[0] if gi.args else {}
        yield {
            "event": GRAPH_INTERRUPT_EVENT,
            "data": data if isinstance(data, dict) else {},
        }


async def finalize_chapter(
    graph: Any,
    config: dict,
    *,
    interrupt_payload: dict[str, Any] | None = None,
    fallback_trace_id: str | None = None,
) -> ChapterOutcome:
    """Read checkpointed state after a stream finishes or interrupts."""
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values) if snapshot and snapshot.values else {}
    next_nodes = tuple(snapshot.next or ()) if snapshot else ()
    trace_id = values.get("trace_id") or fallback_trace_id
    return ChapterOutcome(
        values=values,
        interrupted=bool(next_nodes),
        next_nodes=next_nodes,
        interrupt_payload=interrupt_payload,
        trace_id=trace_id if isinstance(trace_id, str) and trace_id else None,
    )


async def run_chapter(
    graph: Any,
    *,
    config: dict,
    state: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
    on_event: GraphEventCallback | None = None,
    source: str = "api",
) -> ChapterOutcome:
    """Run the chapter graph once. Does not auto-approve human_review."""
    interrupt_payload: dict[str, Any] | None = None
    async with chapter_run_context(
        graph, config=config, state=state, resume=resume, source=source
    ) as ctx:
        async for event in iterate_chapter_events(graph, ctx.payload, ctx.config):
            if event.get("event") == GRAPH_INTERRUPT_EVENT:
                data = event.get("data")
                interrupt_payload = data if isinstance(data, dict) else {}
                continue
            if on_event is not None:
                maybe = on_event(event)
                if inspect.isawaitable(maybe):
                    await maybe
        outcome = await finalize_chapter(
            graph,
            ctx.config,
            interrupt_payload=interrupt_payload,
            fallback_trace_id=ctx.trace_id,
        )
        ctx.handle.record_outcome(outcome.values, interrupted=outcome.interrupted)
        return ChapterOutcome(
            values=outcome.values,
            interrupted=outcome.interrupted,
            next_nodes=outcome.next_nodes,
            interrupt_payload=outcome.interrupt_payload,
            trace_id=outcome.trace_id or ctx.trace_id,
        )


async def run_chapter_until_complete(
    graph: Any,
    state: Mapping[str, Any],
    *,
    config: dict,
    auto_resume: Mapping[str, Any] | None = EVAL_AUTO_APPROVE,
    on_event: GraphEventCallback | None = None,
    max_interrupts: int = 8,
    source: str = "eval",
) -> ChapterOutcome:
    """Run until the graph ends, auto-resuming human_review (eval path)."""
    outcome = await run_chapter(graph, config=config, state=state, on_event=on_event, source=source)
    interrupts = 0
    while outcome.interrupted and auto_resume is not None:
        interrupts += 1
        if interrupts > max_interrupts:
            break
        outcome = await run_chapter(
            graph,
            config=config,
            resume=auto_resume,
            on_event=on_event,
            source=source,
        )
    return outcome
