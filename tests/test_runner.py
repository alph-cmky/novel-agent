"""Tests for the C5 agent loop runner."""

import asyncio

from novel_agent.graph.runner import ChapterOutcome, run_chapter_until_complete


def test_chapter_outcome_dataclass():
    outcome = ChapterOutcome(values={"draft_content": "test"}, interrupted=False)
    assert outcome.values["draft_content"] == "test"
    assert outcome.interrupted is False
    assert outcome.trace_id is None


def test_run_chapter_until_complete_returns_values():
    """The new runner returns a ChapterOutcome with values from run_agent_loop.
    This test verifies the runner interface; actual LLM calls are mocked
    at the integration level.
    """

    class _FakeAgentLoop:
        async def __call__(self, state, **kwargs):
            return {
                "draft_content": "章节正文",
                "editor_report": {"overall_score": 80},
                "human_approved": None,
            }

    # Patch run_agent_loop for this test
    import novel_agent.graph.runner as runner_mod

    original = runner_mod.run_agent_loop
    runner_mod.run_agent_loop = _FakeAgentLoop()
    try:
        outcome = asyncio.run(
            run_chapter_until_complete({"chapter_number": 1, "persist_dir": "/tmp"})
        )
    finally:
        runner_mod.run_agent_loop = original

    assert isinstance(outcome, ChapterOutcome)
    assert outcome.interrupted is False
    assert outcome.values["draft_content"] == "章节正文"
    assert outcome.trace_id  # auto-generated
