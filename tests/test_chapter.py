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
    orchestrator_node,
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
        "evolution_candidates": [
            {
                "version": 1,
                "draft_content": "旧正文",
                "editor_report": {
                    "overall_score": 80,
                    "dimensions": {"rhythm": 100},
                },
                "continuity_report": {"overall_score": 80},
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
        """best dims 只有 1 维时，补零后 dims_avg 分母仍为 5。

        current composite = 80（5 维全 80）。
        修复后 best composite = 80*0.5 + 80*0.3 + (100/5)*0.2 = 68 < 80 → 更新 best。
        修复前 best composite = 80*0.5 + 80*0.3 + (100/1)*0.2 = 84 > 80 → 误判为不更新。
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
                }
            ],
            evolution_best_candidate_version=1,
        )
        result = _run(state)
        # best composite = 60 < 80 → 仍更新 best
        assert result["evolution_best_candidate_version"] == 2


class TestWriterPromptProfileInGraph:
    """P0: graph 节点正确传递 writer_prompt_profile。

    回归测试：
    - orchestrator_node 不再向 OrchestratorAgent.analyze() 传 prompt_profile
      （旧代码会 TypeError: analyze() got an unexpected keyword argument）。
    - writer_node 从 state 读取 writer_prompt_profile 并传入 WriterAgent 构造。
    """

    @staticmethod
    def _orch_state(**overrides) -> dict:
        state = {
            "project_id": "",
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "story_length": "short",
            "character_context": "",
            "world_context": "",
            "skip_orchestrator": False,
        }
        state.update(overrides)
        return state

    def test_orchestrator_does_not_pass_prompt_profile(self):
        """orchestrator_node 不再向 analyze() 传 prompt_profile（修复 TypeError）。"""
        mock_orch = MagicMock()
        mock_orch._compressor = MagicMock()
        mock_orch._compressor.should_compress = MagicMock(return_value=False)
        mock_orch.analyze = AsyncMock(return_value={})
        with patch("novel_agent.graph.chapter.OrchestratorAgent", return_value=mock_orch):
            asyncio.run(orchestrator_node(self._orch_state()))
        kwargs = mock_orch.analyze.call_args.kwargs
        assert "prompt_profile" not in kwargs

    @staticmethod
    def _run_writer_node(state: dict) -> str:
        """Run writer_node, return prompt_profile passed to WriterAgent."""
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured["prompt_profile"] = kwargs.get("prompt_profile")
            mock_w = MagicMock()
            mock_w.write = AsyncMock(
                return_value=("正文" * 600, MagicMock(input_tokens=10, output_tokens=20))
            )
            mock_w.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
            return mock_w

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", side_effect=_capture),
            patch(
                "novel_agent.graph.chapter.QualityService.check_draft_hard_gates",
                return_value={"passed": True, "violations": []},
            ),
            patch(
                "novel_agent.graph.chapter.QualityService.check_story_integrity",
                return_value={"passed": True, "violations": []},
            ),
        ):
            asyncio.run(writer_node(state, None))
        return captured.get("prompt_profile", "")

    def test_writer_node_default_v2(self):
        """state 不含 writer_prompt_profile → 传 v2。"""
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        assert self._run_writer_node(state) == "v2"

    def test_writer_node_explicit_v2(self):
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
            "writer_prompt_profile": "v2",
        }
        assert self._run_writer_node(state) == "v2"

    def test_writer_node_explicit_v1(self):
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
            "writer_prompt_profile": "v1",
        }
        assert self._run_writer_node(state) == "v1"
