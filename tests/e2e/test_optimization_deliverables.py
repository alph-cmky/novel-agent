"""Task-doc §33 final deliverables — measured, not estimated.

Runs the production node functions through the real conditional routers
(graph edges themselves are covered elsewhere) with mocked LLM agents, and
records:

1. reviewer agent constructions by revision type
   (fresh / style-only / logic / world-scope)
2. evolution candidate payload size — storage-backed vs inline-draft baseline
3. writer cost counter accumulation across rounds

No network, no API key required.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from novel_agent.graph.chapter import (
    continuity_node,
    editor_node,
    evolution_orchestrator_node,
    route_after_continuity,
    route_after_editor,
    route_after_writer,
    worldbuilding_node,
    writer_node,
)

STYLE_PLAN = {"focus_dimensions": ["writing", "ai_flavor"], "primary_instruction": "压 AI 味"}
LOGIC_PLAN = {"focus_dimensions": ["consistency"], "primary_instruction": "修因果链"}


def _writer_mock(content="正" * 3000):
    m = MagicMock()
    m.write = AsyncMock(return_value=(content, MagicMock(input_tokens=10, output_tokens=20)))
    m.write_stream = MagicMock()
    m.narrative_extension = AsyncMock(return_value="")
    m.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
    m.model_calls = 1
    m.tool_call_counts = {}
    m.input_tokens = 1000
    m.output_tokens = 500
    m.cached_tokens = 100
    m.reasoning_tokens = 50
    return m


_REVIEWER_TARGETS = {
    "editor": "novel_agent.graph.chapter.EditorAgent",
    "continuity": "novel_agent.graph.chapter.ContinuityAgent",
    "worldbuilding": "novel_agent.graph.chapter.WorldbuildingAgent",
}


class ReviewerCounter:
    """Patch every reviewer node's agent class and count constructions."""

    def __init__(self):
        self.counts = {"editor": 0, "continuity": 0, "worldbuilding": 0}
        self._patches: list = []

    def _make(self, kind: str):
        def _cls(*a, **k):
            self.counts[kind] += 1
            inst = MagicMock()
            if kind == "editor":
                inst.review = AsyncMock(
                    return_value=(
                        {
                            "overall_score": 80,
                            "verdict": "pass",
                            "dimensions": {
                                d: 80
                                for d in ("consistency", "writing", "ai_flavor", "dialogue", "plot", "instruction", "creativity", "controllability")
                            },
                        },
                        MagicMock(),
                    )
                )
                inst.input_tokens = inst.output_tokens = 0
                inst.cached_tokens = inst.reasoning_tokens = 0
            elif kind == "continuity":
                inst.audit = AsyncMock(
                    return_value=({"overall_score": 85, "inconsistencies": []}, MagicMock())
                )
            else:
                inst.extract = AsyncMock(
                    return_value=(
                        {
                            "new_entities": [{"name": "甲", "entity_type": "character"}],
                            "conflicts": [],
                            "foreshadowings": [],
                            "resolved_foreshadowings": [],
                        },
                        MagicMock(),
                    )
                )
            return inst

        return _cls

    def __enter__(self):
        for kind, target in _REVIEWER_TARGETS.items():
            p = patch(target, new=self._make(kind))
            self._patches.append(p)
        profile = MagicMock()
        profile.should_review.return_value = True
        profile.should_worldbuild.return_value = True
        profile.should_enrich_evolution.return_value = False
        self._patches.append(
            patch("novel_agent.graph.chapter.ExecutionProfile.from_state", return_value=profile)
        )
        self._patches.append(patch("novel_agent.graph.chapter._get_chapter_store"))
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


def _next_hop(current: str, state: dict) -> str:
    if current == "evolution_editor":
        return route_after_editor(state)
    if current == "evolution_continuity":
        return route_after_continuity(state)
    return "evolution_orchestrator"


async def _run_round(base_state: dict, plan: dict | None) -> dict:
    """One evolution iteration: writer → production routers → reviews → bookkeeping."""
    state = dict(base_state)
    if plan is not None:
        state["evolution_improvement_plan"] = plan
    state.update(await writer_node(state, None))
    hop = route_after_writer(state)
    while hop in ("evolution_editor", "evolution_continuity", "evolution_worldbuilding"):
        node = {
            "evolution_editor": editor_node,
            "evolution_continuity": continuity_node,
            "evolution_worldbuilding": worldbuilding_node,
        }[hop]
        state.update(await node(state))
        hop = _next_hop(hop, state)
    state.update(await evolution_orchestrator_node(state))
    return state


def _base_state() -> dict:
    return {
        "chapter_number": 5,
        "chapter_outline": "大纲",
        "target_chapter_words": 3000,
        "persist_dir": "./novel-data",
        "project_id": "",
        "context_packet": {},
        "quality_gate_report": {"passed": False},
        "editor_report": {},
        "continuity_report": {},
        "style_report": {},
        "evolution_max_rounds": 3,
    }


def test_deliverable_reviewer_calls_by_revision_type():
    """§33: reviewer 构造次数按 revision type 实测。

    fresh 轮走全链；紧随的 style-only 轮 reviewer 计数不变；
    logic 轮只有 editor+continuity。
    """
    with (
        patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
        patch("novel_agent.graph.chapter.WriterAgent") as w_cls,
        patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as eo_cls,
    ):
        w_cls.return_value = _writer_mock()
        eo_cls.return_value.enrich_plan = AsyncMock(return_value=None)

        with ReviewerCounter() as rc:
            # fresh round: no plan → full chain
            state = asyncio.run(_run_round(_base_state(), None))
            after_fresh = dict(rc.counts)

            # style-only round → zero new reviewers
            state["draft_content"] = "正" * 3001
            state = asyncio.run(_run_round(state, STYLE_PLAN))
            style_delta = {k: rc.counts[k] - after_fresh[k] for k in rc.counts}

            # logic round → editor + continuity only
            before_logic = dict(rc.counts)
            state["draft_content"] = "正" * 3002
            state = asyncio.run(_run_round(state, LOGIC_PLAN))
            logic_delta = {k: rc.counts[k] - before_logic[k] for k in rc.counts}

    print("\n[cost] fresh round reviewers      :", after_fresh)
    print("[cost] style-only round reviewer Δ:", style_delta)
    print("[cost] logic round reviewer Δ     :", logic_delta)
    assert after_fresh == {"editor": 1, "continuity": 1, "worldbuilding": 1}
    assert style_delta == {"editor": 0, "continuity": 0, "worldbuilding": 0}
    assert logic_delta == {"editor": 1, "continuity": 1, "worldbuilding": 0}


def test_deliverable_candidate_checkpoint_size():
    """§39/#8: storage-backed candidate payload vs inline-draft baseline."""
    from novel_agent.graph.chapter import _persist_candidate_draft
    from novel_agent.services.evolution import (
        candidate_from_state,
        composite_score,
        extract_scores,
    )

    async def snapshot(state: dict) -> dict:
        state = dict(state)
        vid = _persist_candidate_draft(state, 1)
        scores = extract_scores(state)
        scores["composite"] = composite_score(scores)
        return candidate_from_state(state, 1, scores, version_id=vid)

    inline_cand = asyncio.run(snapshot({**_base_state(), "draft_content": "长" * 3000}))
    inline_size = len(json.dumps(inline_cand))

    mock_mgr = MagicMock()
    mock_mgr.create_chapter_version.return_value = {"id": "ver-e2e", "content": ""}
    stored_state = {**_base_state(), "project_id": "p1", "draft_content": "长" * 3000}
    with patch("novel_agent.storage.manager.ProjectManager", return_value=mock_mgr):
        stored_cand = asyncio.run(snapshot(stored_state))
    stored_size = len(json.dumps(stored_cand))

    print(f"[size] inline candidate : {inline_size} bytes")
    print(f"[size] storage-backed   : {stored_size} bytes ({stored_size / inline_size:.1%})")
    assert "version_id" in stored_cand and "draft_content" not in stored_cand
    assert stored_size < inline_size * 0.2


def test_deliverable_writer_counters_accumulate():
    """§33: writer 计数跨轮累加（token/model_calls）。"""
    state = _base_state()
    with (
        patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
        patch("novel_agent.graph.chapter.WriterAgent") as w_cls,
    ):
        w_cls.return_value = _writer_mock()
        state.update(asyncio.run(writer_node(state, None)))
        state.update(asyncio.run(writer_node(state, None)))
    print("[cost] writer input tokens / 2 rounds:", state["writer_input_tokens"])
    assert state["writer_input_tokens"] == 2000
