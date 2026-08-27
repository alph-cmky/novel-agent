"""Tests for chapter graph nodes — candidate scores 维度补零。

锁住修复：``evolution_orchestrator_node`` 计算 best_scores 时，若
candidate editor dimensions 缺失某维度，需补 0 到 5 维，否则
``composite_score(best_scores)`` 的 ``dims_avg`` 分母 = 存在的维度数，
与 ``current_scores``（恒 5 维，来自 ``extract_scores``）不一致，
导致 ``is_new_best`` 判断错误。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from novel_agent.graph.chapter import (
    evolution_orchestrator_node,
    route_after_continuity,
    route_after_editor,
    route_after_writer,
    select_best_version_node,
    writer_node,
)


def _state(**overrides) -> dict:
    state = {
        "evolution_round": 1,
        "evolution_version": 2,
        "evolution_history": [
            {
                "v": 1,
                "editor": 70,
                "continuity": 70,
                "composite": 70.0,
                "dimensions": {
                    "rhythm": 70,
                    "ai_flavor": 70,
                    "dialogue": 70,
                    "logic": 70,
                    "writing": 70,
                },
                "style_structure_score": 70,
                "delta": None,
                "focus": None,
            }
        ],
        "editor_report": {
            "overall_score": 80,
            "dimensions": {
                "rhythm": 80,
                "ai_flavor": 80,
                "dialogue": 80,
                "logic": 80,
                "writing": 80,
            },
        },
        "continuity_report": {"overall_score": 80},
        "style_report": {
            "style_gate": "PASS",
            "paragraph_structure_score": 90,
        },
        "evolution_candidates": [
            {
                "version": 1,
                "draft_content": "旧正文",
                "editor_report": {
                    "overall_score": 80,
                    "dimensions": {"rhythm": 100},
                },
                "continuity_report": {"overall_score": 80},
                "style_report": {
                    "style_gate": "PASS",
                    "paragraph_structure_score": 60,
                },
            }
        ],
        "evolution_best_candidate_version": 1,
        "evolution_max_rounds": 5,
        "draft_content": "新正文",
    }
    state.update(overrides)
    return state


def _run(state: dict) -> dict:
    # patch _config_for 避免真实 env/模型路由；patch Agent 类跳过 LLM 调用。
    with (
        patch("novel_agent.graph.chapter._config_for", return_value=None),
        patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
    ):
        mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
        return asyncio.run(evolution_orchestrator_node(state))


class TestBestScoresZeroFill:
    def test_missing_best_dimensions_zero_filled(self):
        """best dims 只有 1 维时，补零后 dim_deltas 仍为 5 维。

        新公式 composite 用 style_structure_score（独立来源），不再依赖 dims_avg。
        current: editor=80, cont=80, style=90 → composite = 82
        best:    editor=80, cont=80, style=60 → composite = 76
        82 > 76 → 更新 best。
        """
        result = _run(_state())
        assert result["evolution_best_candidate_version"] == 2
        assert result["evolution_termination"] == ""

    def test_full_best_dimensions_unchanged(self):
        """best dims 完整 5 维时，补零不改变结果（对照）。"""
        state = _state(
            evolution_candidates=[
                {
                    "version": 1,
                    "draft_content": "旧正文",
                    "editor_report": {
                        "overall_score": 60,
                        "dimensions": {
                            "rhythm": 60,
                            "ai_flavor": 60,
                            "dialogue": 60,
                            "logic": 60,
                            "writing": 60,
                        },
                    },
                    "continuity_report": {"overall_score": 60},
                    "style_report": {
                        "style_gate": "PASS",
                        "paragraph_structure_score": 60,
                    },
                }
            ],
            evolution_best_candidate_version=1,
        )
        result = _run(state)
        # best composite = 60*0.5 + 60*0.3 + 60*0.2 = 60 < 82 → 仍更新 best
        assert result["evolution_best_candidate_version"] == 2


class TestNarrativeExtension:
    """Phase 1: writer_node 用 Narrative Extension 替换 compensation。

    回归测试：
    - actual < target → 触发 narrative_extension（不再调 write 全文重写）
    - extension 是 append，不是 replace
    - actual >= target → 不触发 extension
    - extension 后仍不足 → quality_gate_report['extension_failed'] = True
    - QualityService 最低门槛从 0.85 改为 0.5
    """

    @staticmethod
    def _run_writer_with_mocks(
        write_content: str = "正" * 3000,
        extension_content: str = "",
        target_words: int = 3000,
        model_calls: int = 2,
        tool_call_counts: dict | None = None,
        state_extra: dict | None = None,
    ) -> dict:
        """Run writer_node with mocked WriterAgent, return result dict."""
        mock_writer = MagicMock()
        mock_writer.write = AsyncMock(
            return_value=(write_content, MagicMock(input_tokens=10, output_tokens=20))
        )
        mock_writer.write_stream = MagicMock()
        mock_writer.narrative_extension = AsyncMock(return_value=extension_content)
        mock_writer.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
        mock_writer.model_calls = model_calls
        mock_writer.tool_call_counts = tool_call_counts or {}
        mock_writer.input_tokens = 1000
        mock_writer.output_tokens = 500
        mock_writer.cached_tokens = 200
        mock_writer.reasoning_tokens = 50

        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": target_words,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        if state_extra:
            state.update(state_extra)

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", return_value=mock_writer),
        ):
            result = asyncio.run(writer_node(state, None))
        return result

    def test_short_content_triggers_extension(self):
        """actual < target → 调用 narrative_extension。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1500,
            extension_content="续" * 1500,
            target_words=3000,
        )
        assert "续" * 1500 in result["draft_content"]
        assert "短" * 1500 in result["draft_content"]

    def test_extension_appends_not_replaces(self):
        """Extension 是 append，原始内容保留。"""
        result = self._run_writer_with_mocks(
            write_content="原始内容",
            extension_content="续写内容",
            target_words=3000,
        )
        assert "原始内容" in result["draft_content"]
        assert "续写内容" in result["draft_content"]

    def test_target_length_no_extension(self):
        """actual >= target → 不触发 extension。"""
        mock_writer = MagicMock()
        mock_writer.write = AsyncMock(
            return_value=("正" * 3000, MagicMock(input_tokens=10, output_tokens=20))
        )
        mock_writer.narrative_extension = AsyncMock(return_value="不该出现")
        mock_writer.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
        mock_writer.model_calls = 1
        mock_writer.tool_call_counts = {}
        mock_writer.input_tokens = 1000
        mock_writer.output_tokens = 500
        mock_writer.cached_tokens = 200
        mock_writer.reasoning_tokens = 50

        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 3000,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", return_value=mock_writer),
        ):
            result = asyncio.run(writer_node(state, None))
        mock_writer.narrative_extension.assert_not_called()
        assert "不该出现" not in result["draft_content"]

    def test_extension_still_short_sets_failed(self):
        """Extension 后仍不足 → extension_failed + passed=False + violation。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1000,
            extension_content="续" * 500,
            target_words=3000,
        )
        gate = result["quality_gate_report"]
        assert gate.get("extension_failed") is True
        assert gate["passed"] is False
        assert "length_target_unmet" in gate["violations"]

    def test_extension_success_no_failed_flag(self):
        """Extension 后达标 → 不设 extension_failed。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1500,
            extension_content="续" * 1500,
            target_words=3000,
        )
        assert "extension_failed" not in result["quality_gate_report"]

    def test_extension_trigger_boundaries(self):
        """Phase G: 触发边界 — 2999 触发 / 3000 不触发 / 3300 不触发。"""
        from novel_agent.graph.chapter import _should_extend

        assert _should_extend(1500, 3000) is True  # 半篇幅 → 续写
        assert _should_extend(2500, 3000) is True  # 明显不足 → 续写
        assert _should_extend(2999, 3000) is True  # 差 1 字 → 仍续写
        assert _should_extend(3000, 3000) is False  # 达标 → 不续写
        assert _should_extend(3300, 3000) is False  # 超标 → 不续写
        assert _should_extend(3450, 3000) is False  # 超标 → 不续写
        assert _should_extend(5000, 3000) is False  # 大幅超标 → 不续写
        # 目标过小（<1000）不启用 extension 机制
        assert _should_extend(500, 800) is False

    def test_over_target_content_never_truncated(self):
        """Phase G: 超过 target（>115%）的内容不被强制截断。"""
        over = "超" * 3600  # 3600/3000 = 120%
        result = self._run_writer_with_mocks(
            write_content=over,
            extension_content="",
            target_words=3000,
        )
        assert result["draft_content"] == over
        assert "extension_failed" not in result["quality_gate_report"]

    def test_quality_gate_uses_half_target_not_85(self):
        """QualityService 硬性检查最低门槛从 0.85 改为 0.5。"""
        from novel_agent.services.quality import QualityService

        # 50% of target → 0.85 时代会 fail，0.5 时代 pass
        report = QualityService.check_draft_hard_gates(
            "字" * 1500,
            target_words=3000,
            chapter_outline="大纲",
        )
        assert report["passed"] is True

        # 40% of target → 0.5 以下，应该 fail
        report = QualityService.check_draft_hard_gates(
            "字" * 1200,
            target_words=3000,
            chapter_outline="大纲",
        )
        assert "minimum_length" in report["violations"]


class TestWriterCostCounters:
    """Phase J: writer tool-loop 成本观测（不动 max_rounds，只计数）。

    writer_model_calls / writer_tool_calls / writer_search_calls 从
    WriterAgent 实例计数器累计到 state，跨进化轮次累加。
    """

    def test_counters_from_agent_instance(self):
        """model=3, tools={search:2, other:1} → 3/3/2。"""
        result = TestNarrativeExtension._run_writer_with_mocks(
            model_calls=3,
            tool_call_counts={"search_context": 2, "other_tool": 1},
        )
        assert result["writer_model_calls"] == 3
        assert result["writer_tool_calls"] == 3
        assert result["writer_search_calls"] == 2

    def test_counters_zero_without_tool_use(self):
        """无工具调用 → tool/search 计数为 0。"""
        result = TestNarrativeExtension._run_writer_with_mocks(
            model_calls=1,
            tool_call_counts={},
        )
        assert result["writer_model_calls"] == 1
        assert result["writer_tool_calls"] == 0
        assert result["writer_search_calls"] == 0

    def test_counters_accumulate_across_rounds(self):
        """state 已有计数 → 本轮 agent 计数累加，不覆盖。"""
        result = TestNarrativeExtension._run_writer_with_mocks(
            model_calls=2,
            tool_call_counts={"search_context": 1},
            state_extra={
                "writer_model_calls": 3,
                "writer_tool_calls": 2,
                "writer_search_calls": 1,
            },
        )
        assert result["writer_model_calls"] == 5
        assert result["writer_tool_calls"] == 3
        assert result["writer_search_calls"] == 2

    def test_token_usage_from_agent_instance(self):
        """Phase 7: 四类 token 从实例计数器写入 state。"""
        result = TestNarrativeExtension._run_writer_with_mocks()
        assert result["writer_input_tokens"] == 1000
        assert result["writer_output_tokens"] == 500
        assert result["writer_cached_tokens"] == 200
        assert result["writer_reasoning_tokens"] == 50

    def test_token_usage_accumulates_across_rounds(self):
        """跨进化轮次 token 累加，不覆盖。"""
        result = TestNarrativeExtension._run_writer_with_mocks(
            state_extra={
                "writer_input_tokens": 100,
                "writer_output_tokens": 50,
                "writer_cached_tokens": 20,
                "writer_reasoning_tokens": 5,
            },
        )
        assert result["writer_input_tokens"] == 1100
        assert result["writer_output_tokens"] == 550
        assert result["writer_cached_tokens"] == 220
        assert result["writer_reasoning_tokens"] == 55


class TestStyleAnalyzerRunsAfterWriter:
    """Phase 1 (P0-1): StyleAnalyzer must run on every draft, not only when Editor runs.

    Regression: when route_after_writer skips Editor (deterministic_gate_first +
    gate PASS), style_report was never produced, so extract_scores defaulted
    style_structure_score to 100 — treating "unanalyzed" as "perfect".
    """

    @staticmethod
    def _run_writer_with_mocks(write_content: str = "正" * 3000) -> dict:
        mock_writer = MagicMock()
        mock_writer.write = AsyncMock(
            return_value=(write_content, MagicMock(input_tokens=10, output_tokens=20))
        )
        mock_writer.write_stream = MagicMock()
        mock_writer.narrative_extension = AsyncMock(return_value="")
        mock_writer.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
        mock_writer.model_calls = 1
        mock_writer.tool_call_counts = {}
        mock_writer.input_tokens = 100
        mock_writer.output_tokens = 50
        mock_writer.cached_tokens = 0
        mock_writer.reasoning_tokens = 0

        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 3000,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", return_value=mock_writer),
        ):
            return asyncio.run(writer_node(state, None))

    def test_style_analyzer_runs_after_writer(self):
        """writer_node output includes style_report from deterministic StyleAnalyzer."""
        result = self._run_writer_with_mocks()
        assert "style_report" in result
        assert result["style_report"]
        assert "style_gate" in result["style_report"]
        assert "paragraph_structure" in result["style_report"]

    def test_style_report_persisted_in_state(self):
        """style_report contains real metrics from StyleAnalyzer, not empty dict."""
        result = self._run_writer_with_mocks()
        sr = result["style_report"]
        assert "paragraph_structure_score" in sr
        assert isinstance(sr["paragraph_structure_score"], (int, float))
        assert "ai_flavor_score" in sr

    def test_style_report_present_when_quality_gate_skips_editor(self):
        """style_report is in writer_node output even when Editor is skipped.

        route_after_writer reads quality_gate_report from writer_node output;
        style_report is also in writer_node output, so it enters state
        before the routing decision — regardless of whether Editor runs.
        """
        result = self._run_writer_with_mocks()
        # writer_node must always produce style_report, independent of gate result
        assert "style_report" in result
        assert result["style_report"]
        # quality_gate_report is also present (from the same writer_node)
        assert "quality_gate_report" in result

    def test_missing_style_report_is_not_score_100(self):
        """extract_scores must not default style_structure_score to 100 when missing."""
        from novel_agent.services.evolution import extract_scores

        # State with no style_report — must NOT yield score 100
        scores = extract_scores({"editor_report": {"overall_score": 80}})
        assert scores["style_structure_score"] == 0
        assert scores["style_gate"] == "PASS"  # gate defaults to PASS (no anomaly detected)


class TestEvolutionConditionalRouting:
    """Phase 5 (P1-3): revision scope determines reviewers — style-only skips LLM review."""

    @staticmethod
    def _plan(focus: list[str], instruction: str = "改进") -> dict:
        return {"focus_dimensions": focus, "primary_instruction": instruction}

    # ── required_reviewers (deterministic service rule) ──

    def test_required_reviewers_conservative_without_focus(self):
        """空 focus（新章/人类拒绝/未知 plan）→ 全量保守执行。"""
        from novel_agent.services.evolution import required_reviewers

        full = {"editor": True, "continuity": True, "worldbuilding": True}
        assert required_reviewers({}) == full
        assert required_reviewers(None) == full
        reject = {"focus_dimensions": [], "primary_instruction": "人类审阅者拒绝了这个版本"}
        assert required_reviewers(reject) == full

    def test_required_reviewers_style_only_skips_all(self):
        from novel_agent.services.evolution import required_reviewers

        r = required_reviewers(self._plan(["rhythm", "ai_flavor"]))
        assert r == {"editor": False, "continuity": False, "worldbuilding": False}

    def test_required_reviewers_logic_scope_runs_editor_continuity_only(self):
        from novel_agent.services.evolution import required_reviewers

        r = required_reviewers(self._plan(["logic"]))
        assert r["editor"] is True
        assert r["continuity"] is True
        assert r["worldbuilding"] is False

    def test_required_reviewers_world_keywords_rerun_worldbuilding(self):
        from novel_agent.services.evolution import required_reviewers

        r = required_reviewers(self._plan(["logic"], instruction="修正世界观：北墙势力归属变更"))
        assert r["worldbuilding"] is True

    def test_required_reviewers_unknown_dimension_keeps_worldbuilding(self):
        """非 Editor 维度无法判断 → 保守跑 Worldbuilding。"""
        from novel_agent.services.evolution import required_reviewers

        r = required_reviewers(self._plan(["worldbuilding_consistency"]))
        assert r["worldbuilding"] is True

    # ── graph routers ──

    def test_route_after_writer_style_only_goes_to_orchestrator(self):
        state = {
            "quality_gate_report": {"passed": False},
            "chapter_number": 3,
            "evolution_improvement_plan": self._plan(["rhythm"]),
        }
        assert route_after_writer(state) == "evolution_orchestrator"

    def test_route_after_writer_full_plan_goes_to_editor(self):
        state = {
            "quality_gate_report": {"passed": False},
            "chapter_number": 3,
            "evolution_improvement_plan": self._plan([]),
        }
        assert route_after_writer(state) == "evolution_editor"

    def test_route_after_editor_logic_scope_keeps_continuity(self):
        state = {"evolution_improvement_plan": self._plan(["logic"])}
        assert route_after_editor(state) == "evolution_continuity"

    def test_route_after_continuity_logic_scope_skips_worldbuilding(self):
        state = {"evolution_improvement_plan": self._plan(["logic"])}
        assert route_after_continuity(state) == "evolution_orchestrator"

    def test_route_after_continuity_world_scope_runs_worldbuilding(self):
        state = {
            "evolution_improvement_plan": self._plan(
                ["dialogue"], instruction="调整实体关系与设定冲突"
            )
        }
        assert route_after_continuity(state) == "evolution_worldbuilding"

    def test_fresh_chapter_without_plan_runs_full_chain(self):
        state = {}
        assert route_after_editor(state) == "evolution_continuity"
        assert route_after_continuity(state) == "evolution_worldbuilding"


class TestCandidateStateFootprint:
    """Phase 6 (P1-4): storage-backed candidates keep version_id, not full drafts."""

    @staticmethod
    def _storage_state(**overrides) -> dict:
        state = _state()
        state.update(
            {
                "project_id": "p1",
                "persist_dir": "./novel-data",
                "writing_run_id": "run-1",
                "chapter_number": 5,
                "draft_content": "新正文" * 100,
            }
        )
        state.update(overrides)
        return state

    def test_persisted_candidate_stores_version_id_not_draft(self):
        mock_mgr = MagicMock()
        mock_mgr.create_chapter_version.return_value = {"id": "ver-abc", "content": "x"}
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=None),
            patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
            patch("novel_agent.storage.manager.ProjectManager", return_value=mock_mgr),
        ):
            mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
            result = asyncio.run(evolution_orchestrator_node(self._storage_state()))

        candidate = result["evolution_candidates"][-1]
        assert candidate["version_id"] == "ver-abc"
        assert "draft_content" not in candidate
        assert candidate["content_length"] == len("新正文" * 100)
        # one INSERT per round, tagged with the evolution origin
        call = mock_mgr.create_chapter_version.call_args
        assert call.kwargs["origin"] == "evolution_v2"

    def test_persistence_failure_falls_back_to_inline_draft(self):
        mock_mgr = MagicMock()
        mock_mgr.create_chapter_version.side_effect = RuntimeError("db down")
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=None),
            patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
            patch("novel_agent.storage.manager.ProjectManager", return_value=mock_mgr),
        ):
            mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
            result = asyncio.run(evolution_orchestrator_node(self._storage_state()))

        candidate = result["evolution_candidates"][-1]
        assert candidate["draft_content"] == "新正文" * 100
        assert "version_id" not in candidate

    def test_no_project_id_skips_persistence_and_keeps_inline_draft(self):
        state = self._storage_state(project_id="", persist_dir="")
        result = asyncio.run(evolution_orchestrator_node(state))
        candidate = result["evolution_candidates"][-1]
        assert candidate["draft_content"]
        assert "version_id" not in candidate

    def test_select_best_rollback_loads_draft_from_storage(self):
        state = _state()
        state.update({"project_id": "p1", "persist_dir": "./novel-data"})
        best = {
            "version": 1,
            "version_id": "ver-old",
            "editor_report": {"overall_score": 80, "dimensions": {"rhythm": 80}},
            "continuity_report": {"overall_score": 80},
            "style_report": {"style_gate": "PASS", "paragraph_structure_score": 60},
            "worldbuilding_report": {},
            "quality_gate_report": {},
            "outline_coverage": None,
            "required_facts_missing": 0,
            "content_length": 4,
        }
        state["evolution_candidates"] = [best]

        mock_mgr = MagicMock()
        mock_mgr.get_chapter_version.return_value = {
            "id": "ver-old",
            "content": "旧正文全文",
        }
        with patch("novel_agent.storage.manager.ProjectManager", return_value=mock_mgr):
            result = select_best_version_node(state)

        assert result["draft_content"] == "旧正文全文"
        mock_mgr.get_chapter_version.assert_called_once_with("ver-old")

    def test_select_best_rollback_without_storage_access_returns_empty_draft(self):
        """无 project_id 时 loader 为 None → 不崩溃，draft 留空（SSE 兜底走当前文）。"""
        state = _state()
        state["project_id"] = ""
        state["evolution_candidates"] = [
            {"version": 1, "version_id": "ver-old", "editor_report": {}}
        ]
        result = select_best_version_node(state)
        assert result["draft_content"] == ""


class TestEvolutionContextMinimal:
    """Task 4: evolution enrichment 默认不调用 + 不传完整 draft。"""

    @staticmethod
    def _run_evo_first_round(state_extra: dict | None = None) -> dict:
        """Run evolution_orchestrator_node first-round branch."""
        state = {
            "evolution_round": 0,
            "evolution_version": 0,
            "evolution_history": [],
            "editor_report": {"overall_score": 70, "dimensions": {}},
            "continuity_report": {"overall_score": 70},
            "evolution_max_rounds": 5,
        }
        if state_extra:
            state.update(state_extra)
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=None),
            patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
        ):
            mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
            result = asyncio.run(evolution_orchestrator_node(state))
        return result, mock_agent

    def test_first_round_enrichment_not_called_when_rule_plan_has_instruction(self):
        """规则计划有 primary_instruction 时不调用 LLM enrichment。"""
        result, mock_agent = self._run_evo_first_round()
        mock_agent.return_value.enrich_plan.assert_not_called()

    def test_first_round_enrichment_called_when_rule_plan_empty(self):
        """规则计划缺 primary_instruction 时才调用 LLM enrichment。"""
        from unittest.mock import patch as _patch

        with _patch(
            "novel_agent.graph.chapter.build_improvement_plan_rule",
            return_value={"primary_instruction": "", "focus_dimensions": []},
        ):
            result, mock_agent = self._run_evo_first_round({"skip_evolution_enrichment": False})
        mock_agent.return_value.enrich_plan.assert_called_once()

    def test_counters_zero_enrichment_normal_chapter(self):
        """Phase I: 正常章节 rule plan 有 instruction → rule=1, enrichment=0。"""
        result, _ = self._run_evo_first_round()
        assert result["evolution_rule_plan_calls"] == 1
        assert result["evolution_llm_enrichment_calls"] == 0

    def test_counters_enrichment_counted_when_called(self):
        """Phase I: rule plan 缺 instruction → enrichment 计 1 次。"""
        from unittest.mock import patch as _patch

        with _patch(
            "novel_agent.graph.chapter.build_improvement_plan_rule",
            return_value={"primary_instruction": "", "focus_dimensions": []},
        ):
            result, _ = self._run_evo_first_round({"skip_evolution_enrichment": False})
        assert result["evolution_rule_plan_calls"] == 1
        assert result["evolution_llm_enrichment_calls"] == 1

    def test_counters_accumulate_across_rounds(self):
        """Phase I: 计数器从 state 既有值累加，不覆盖。"""
        result, _ = self._run_evo_first_round(
            {"evolution_rule_plan_calls": 2, "evolution_llm_enrichment_calls": 1}
        )
        assert result["evolution_rule_plan_calls"] == 3
        assert result["evolution_llm_enrichment_calls"] == 1

    @staticmethod
    def test_enrichment_receives_violations_not_draft():
        """Phase H: enrich_plan 只收 violations，不收 draft_preview/正文。

        Evolution 是元评估器——上下文仅限 scores/delta/violations/
        improvement_plan，正文泄漏进 enrichment prompt 属于回归。
        """
        from unittest.mock import patch as _patch

        long_draft = "草" * 5000
        with _patch(
            "novel_agent.graph.chapter.build_improvement_plan_rule",
            return_value={"primary_instruction": "", "focus_dimensions": []},
        ):
            state = {
                "evolution_round": 0,
                "evolution_version": 0,
                "evolution_history": [],
                "editor_report": {"overall_score": 70, "dimensions": {}},
                "continuity_report": {"overall_score": 70},
                "evolution_max_rounds": 5,
                "draft_content": long_draft,
                "skip_evolution_enrichment": False,
            }
            with (
                patch("novel_agent.graph.chapter._config_for", return_value=None),
                patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
            ):
                mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
                asyncio.run(evolution_orchestrator_node(state))

        call_kwargs = mock_agent.return_value.enrich_plan.call_args.kwargs
        assert "draft_preview" not in call_kwargs
        assert "violations" in call_kwargs
        assert isinstance(call_kwargs["violations"], list)


class TestContextMinimality:
    """Graph-level: verify agents receive minimal context, not full State.

    Regression guard against re-introducing full-context leaks.
    """

    @staticmethod
    def _editor_state(**overrides) -> dict:
        state = {
            "chapter_number": 1,
            "draft_content": "正文内容" * 100,
            "narrative_mode": None,
            "style_report": {
                "style_gate": "PASS",
                "paragraph_structure_score": 90,
                "paragraph_structure": {"paragraph_count": 10},
            },
            "context_packet": {
                "character_context": "角色" * 5000,
                "world_context": "设定" * 5000,
                "recent_summary": "摘要" * 5000,
                "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
                "timeline_events": [{"e": f"事件{i}"} for i in range(30)],
                "timeline_findings": [{"f": f"发现{i}"} for i in range(10)],
            },
        }
        state.update(overrides)
        return state

    def test_orchestrator_node_plans_on_minimal_projection(self):
        """Phase 2: Orchestrator 收 for_orchestrator 投影，world_context 有界。"""
        from novel_agent.graph.chapter import orchestrator_node

        captured: dict = {}

        async def _mock_analyze(**kwargs):
            captured.update(kwargs)
            return {"narrative_stage": "development", "chapter_strategy": {}}

        state = {
            "chapter_number": 2,
            "chapter_outline": "大纲",
            "context_packet": {
                "character_context": "角色" * 5000,
                "world_context": "设定" * 5000,
                "recent_summary": "摘要" * 5000,
                "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
                "timeline_events": [{"e": f"事件{i}"} for i in range(30)],
                "timeline_findings": [{"f": f"发现{i}"} for i in range(10)],
            },
        }
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.OrchestratorAgent") as mock_cls,
        ):
            mock_cls.return_value.analyze = _mock_analyze
            mock_cls.return_value.input_tokens = 3000
            mock_cls.return_value.output_tokens = 800
            mock_cls.return_value.cached_tokens = 100
            mock_cls.return_value.reasoning_tokens = 0
            result = asyncio.run(orchestrator_node(state))

        packet = captured["context_packet"]
        # for_orchestrator: world 1/4 预算、伏笔 ≤10、事件 ≤8
        assert len(packet["world_context"]) <= 4000 // 4 + 30
        assert len(packet["character_context"]) <= 4000 // 2 + 30
        assert len(packet["unresolved_foreshadowings"]) <= 10
        assert len(packet["timeline_events"]) <= 8
        # Phase 7: token 观测写入 state
        assert result["orchestrator_input_tokens"] == 3000
        assert result["orchestrator_output_tokens"] == 800
        assert result["orchestrator_cached_tokens"] == 100
        assert result["orchestrator_reasoning_tokens"] == 0

    def test_editor_node_passes_style_report_and_minimal_context(self):
        """Editor receives StyleReport from state + for_editor projection."""
        from novel_agent.graph.chapter import editor_node

        captured: dict = {}

        async def _mock_review(**kwargs):
            captured.update(kwargs)
            return {"overall_score": 80, "verdict": "pass"}, MagicMock()

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.EditorAgent") as mock_editor_cls,
        ):
            mock_editor_cls.return_value.review = _mock_review
            mock_editor_cls.return_value.input_tokens = 2000
            mock_editor_cls.return_value.output_tokens = 300
            mock_editor_cls.return_value.cached_tokens = 0
            mock_editor_cls.return_value.reasoning_tokens = 0
            result = asyncio.run(editor_node(self._editor_state()))

        # style_report is passed through from state, not recomputed
        assert "style_report" in captured
        assert captured["style_report"] is not None
        assert captured["style_report"]["style_gate"] == "PASS"
        assert "paragraph_structure" in captured["style_report"]

        # context_packet must be the minimal for_editor projection
        ctx = captured.get("context_packet") or {}
        assert "world_context" not in ctx
        assert "timeline_events" not in ctx
        assert len(ctx.get("unresolved_foreshadowings", [])) <= 3

        # Phase 7: token 观测写入 state
        assert result["editor_input_tokens"] == 2000
        assert result["editor_output_tokens"] == 300

    def test_editor_node_returns_style_report_in_state(self):
        """editor_node passes style_report through from state for Evolution consumption."""
        from novel_agent.graph.chapter import editor_node

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.EditorAgent") as mock_editor_cls,
        ):
            mock_editor_cls.return_value.review = AsyncMock(
                return_value=({"overall_score": 80, "verdict": "pass"}, MagicMock())
            )
            mock_editor_cls.return_value.input_tokens = 0
            mock_editor_cls.return_value.output_tokens = 0
            mock_editor_cls.return_value.cached_tokens = 0
            mock_editor_cls.return_value.reasoning_tokens = 0
            result = asyncio.run(editor_node(self._editor_state()))

        assert "style_report" in result
        assert result["style_report"]
        assert "style_gate" in result["style_report"]

    def test_editor_node_unavailable_still_records_tokens(self):
        """unavailable 早退分支同样回写 token（重试消耗不丢失）。"""
        from novel_agent.graph.chapter import editor_node

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.EditorAgent") as mock_editor_cls,
        ):
            mock_editor_cls.return_value.review = AsyncMock(
                return_value=({"unavailable": True}, MagicMock())
            )
            mock_editor_cls.return_value.input_tokens = 600
            mock_editor_cls.return_value.output_tokens = 0
            mock_editor_cls.return_value.cached_tokens = 0
            mock_editor_cls.return_value.reasoning_tokens = 0
            result = asyncio.run(editor_node(self._editor_state()))

        assert result["editor_report"].get("unavailable") is True
        assert result["editor_input_tokens"] == 600
        assert result["editor_output_tokens"] == 0

    @staticmethod
    def _continuity_state(**overrides) -> dict:
        state = {
            "chapter_number": 1,
            "draft_content": "正文内容" * 100,
            "narrative_mode": None,
            "persist_dir": "./novel-data",
            "project_id": "",
            "context_packet": {
                "character_context": "角色" * 5000,
                "world_context": "设定" * 5000,
                "recent_summary": "摘要" * 5000,
                "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
                "timeline_events": [{"e": f"事件{i}"} for i in range(30)],
                "timeline_findings": [{"f": f"发现{i}"} for i in range(10)],
            },
        }
        state.update(overrides)
        return state

    def test_continuity_node_passes_minimal_context(self):
        """Continuity receives for_continuity projection, not full packet."""
        from novel_agent.graph.chapter import continuity_node

        captured: dict = {}

        async def _mock_audit(**kwargs):
            captured.update(kwargs)
            return {"overall_score": 90, "inconsistencies": []}, MagicMock()

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.ContinuityAgent") as mock_cty_cls,
        ):
            mock_cty_cls.return_value.audit = _mock_audit
            asyncio.run(continuity_node(self._continuity_state()))

        ctx = captured.get("context_packet") or {}
        # for_continuity keeps timeline_events but not world_context or recent_summary
        assert "world_context" not in ctx
        assert "recent_summary" not in ctx
        assert len(ctx.get("timeline_events", [])) <= 10
        assert len(ctx.get("unresolved_foreshadowings", [])) <= 8
