"""Single chapter-execution entry for API, CLI, and eval.

Wraps the C5 agent loop (Orchestrator → Writer loop → Editor → Continuity →
Worldbuilding) and returns a ChapterOutcome compatible with the API layer.
No LangGraph, no checkpoint interrupts — human review is a post-generation step.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from novel_agent.graph.agent_loop import run_agent_loop
from novel_agent.observability.tracing import chapter_trace

GraphEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class ChapterOutcome:
    """Result of one agent-loop invocation."""

    values: dict[str, Any]
    interrupted: bool = False
    next_nodes: tuple[str, ...] = ()
    interrupt_payload: dict[str, Any] | None = None
    trace_id: str | None = None


async def run_chapter_agent_loop(
    state: Mapping[str, Any],
    *,
    on_event: GraphEventCallback | None = None,
    source: str = "api",
    max_rounds: int = 8,
    max_retries: int = 2,
) -> ChapterOutcome:
    """Run the C5 agent loop and return a ChapterOutcome.

    The agent loop runs to completion (no interrupts). Human review is handled
    by the API layer after this returns, as a separate approve/reject step.
    """
    payload = dict(state)
    trace_id = payload.get("trace_id") or uuid.uuid4().hex
    payload["trace_id"] = trace_id

    async with chapter_trace(payload, source=source) as handle:
        if handle.trace_id:
            payload["trace_id"] = handle.trace_id
            trace_id = handle.trace_id

        if on_event:
            maybe = on_event({"event": "agent_loop_start", "data": payload})
            if inspect_awaitable(maybe):
                await maybe

        values = await run_agent_loop(payload, max_rounds=max_rounds, max_retries=max_retries)
        values["trace_id"] = trace_id

        handle.record_outcome(values, interrupted=False)

        if on_event:
            maybe = on_event({"event": "agent_loop_done", "data": values})
            if inspect_awaitable(maybe):
                await maybe

    return ChapterOutcome(values=values, interrupted=False, trace_id=trace_id)


async def run_chapter_until_complete(
    state: Mapping[str, Any],
    *,
    on_event: GraphEventCallback | None = None,
    source: str = "eval",
    max_rounds: int = 8,
    max_retries: int = 2,
) -> ChapterOutcome:
    """Eval path: run the agent loop to completion (no human review interrupt)."""
    return await run_chapter_agent_loop(
        state,
        on_event=on_event,
        source=source,
        max_rounds=max_rounds,
        max_retries=max_retries,
    )


def inspect_awaitable(maybe: Any) -> bool:
    import inspect

    return inspect.isawaitable(maybe)
