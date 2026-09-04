"""Tests for the shared chapter graph runner."""

import asyncio
from types import SimpleNamespace

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from novel_agent.graph.runner import (
    chapter_payload,
    run_chapter,
    run_chapter_until_complete,
)


class _FakeGraph:
    def __init__(self, events=(), snapshot=None, interrupt=None):
        self.events = list(events)
        self.snapshot = snapshot or SimpleNamespace(next=(), values={})
        self.interrupt = interrupt
        self.inputs: list = []

    async def astream_events(self, payload, _config, version):
        self.inputs.append(payload)
        if self.interrupt is not None:
            raise self.interrupt
        for event in self.events:
            yield event

    async def aget_state(self, _config):
        return self.snapshot


def test_chapter_payload_assigns_trace_id():
    payload = chapter_payload(state={"chapter_number": 2})
    assert isinstance(payload, dict)
    assert payload["chapter_number"] == 2
    assert payload["trace_id"]


def test_chapter_payload_keeps_existing_trace_id():
    payload = chapter_payload(state={"trace_id": "fixed-id"})
    assert payload["trace_id"] == "fixed-id"


def test_chapter_payload_resume_is_command():
    payload = chapter_payload(resume={"action": "approve", "comments": ""})
    assert isinstance(payload, Command)


def test_run_chapter_reads_checkpoint_and_interrupt():
    snapshot = SimpleNamespace(
        next=("human_review",),
        values={"draft_content": "候选", "trace_id": "t1"},
    )
    graph = _FakeGraph(snapshot=snapshot)
    outcome = asyncio.run(run_chapter(graph, config={}, state={"writing_run_id": "r1"}))

    assert outcome.interrupted is True
    assert outcome.next_nodes == ("human_review",)
    assert outcome.values["draft_content"] == "候选"
    assert outcome.trace_id == "t1"
    assert graph.inputs[0]["writing_run_id"] == "r1"
    assert graph.inputs[0]["trace_id"]


def test_run_chapter_surfaces_graph_interrupt_payload():
    snapshot = SimpleNamespace(next=("human_review",), values={})
    graph = _FakeGraph(
        snapshot=snapshot,
        interrupt=GraphInterrupt({"chapter_content": "稿"}),
    )
    outcome = asyncio.run(run_chapter(graph, config={}, state={}))
    assert outcome.interrupt_payload == {"chapter_content": "稿"}


def test_run_chapter_until_complete_auto_approves():
    class _ResumeGraph:
        def __init__(self):
            self.calls = 0

        async def astream_events(self, _payload, _config, version):
            if False:
                yield

        async def aget_state(self, _config):
            self.calls += 1
            if self.calls <= 2:
                return SimpleNamespace(next=("human_review",), values={"draft_content": "稿"})
            return SimpleNamespace(
                next=(),
                values={"draft_content": "稿", "human_approved": True},
            )

    graph = _ResumeGraph()
    outcome = asyncio.run(run_chapter_until_complete(graph, {"chapter_number": 1}, config={}))
    assert outcome.interrupted is False
    assert outcome.values["human_approved"] is True
    assert graph.calls == 3
