"""Chapter tracing: default off; resume keeps the same trace_id."""

import asyncio
from types import SimpleNamespace

import pytest

from novel_agent.graph.runner import run_chapter
from novel_agent.observability.tracing import (
    chapter_trace,
    require_tracing_config,
)


def test_tracing_off_yields_null_handle(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)

    async def _run():
        async with chapter_trace({"chapter_number": 1, "trace_id": "abc"}) as handle:
            assert handle.enabled is False
            assert handle.handler is None
            assert handle.trace_id == "abc"
            handle.record_outcome({"draft_content": "x"})
            handle.event("noop")
            with handle.tool_span("search", {"query": "q"}) as span:
                span.update(output={"ok": True})

    asyncio.run(_run())


def test_require_tracing_config_strict_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LANGFUSE_"):
        require_tracing_config(strict=True)


def test_require_tracing_config_eval_returns_message(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    msg = require_tracing_config(strict=False)
    assert msg
    assert "LANGFUSE_" in msg


def test_require_tracing_off_is_silent(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "0")
    assert require_tracing_config(strict=True) is None


def test_resume_reuses_trace_id(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)

    class _Graph:
        def __init__(self):
            self.values: dict = {}

        async def astream_events(self, payload, _config, version):
            if isinstance(payload, dict) and payload.get("trace_id"):
                self.values["trace_id"] = payload["trace_id"]
            if False:
                yield

        async def aget_state(self, _config):
            return SimpleNamespace(next=(), values=dict(self.values))

    graph = _Graph()
    first = asyncio.run(run_chapter(graph, config={}, state={"chapter_number": 2}))
    second = asyncio.run(
        run_chapter(graph, config={}, resume={"action": "approve", "comments": ""})
    )
    assert first.trace_id
    assert first.trace_id == second.trace_id
    assert first.trace_id == graph.values["trace_id"]
