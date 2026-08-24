"""Tests for chapter graph nodes — candidate scores 维度补零。

锁住修复：``evolution_orchestrator_node`` 计算 best_scores 时，若
candidate editor dimensions 缺失某维度，需补 0 到 5 维，否则
``composite_score(best_scores)`` 的 ``dims_avg`` 分母 = 存在的维度数，
与 ``current_scores``（恒 5 维，来自 ``extract_scores``）不一致，
导致 ``is_new_best`` 判断错误。
"""

import asyncio
from unittest.mock import AsyncMock, patch

from novel_agent.graph.chapter import evolution_orchestrator_node


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
                    "rhythm": 70, "ai_flavor": 70, "dialogue": 70,
                    "logic": 70, "writing": 70,
                },
                "delta": None,
                "focus": None,
            }
        ],
        "editor_report": {
            "overall_score": 80,
            "dimensions": {
                "rhythm": 80, "ai_flavor": 80, "dialogue": 80,
                "logic": 80, "writing": 80,
            },
        },
        "continuity_report": {"overall_score": 80},
        "evolution_candidates": [{
            "version": 1,
            "draft_content": "旧正文",
            "editor_report": {
                "overall_score": 80, "dimensions": {"rhythm": 100},
            },
            "continuity_report": {"overall_score": 80},
        }],
        "evolution_best_candidate_version": 1,
        "evolution_max_rounds": 5,
        "draft_content": "新正文",
    }
    state.update(overrides)
    return state


def _run(state: dict) -> dict:
    # patch _config_for 避免真实 env/模型路由；patch Agent 类跳过 LLM 调用。
    with patch("novel_agent.graph.chapter._config_for", return_value=None), \
         patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent:
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
            evolution_candidates=[{
                "version": 1,
                "draft_content": "旧正文",
                "editor_report": {
                    "overall_score": 60,
                    "dimensions": {
                        "rhythm": 60, "ai_flavor": 60, "dialogue": 60,
                        "logic": 60, "writing": 60,
                    },
                },
                "continuity_report": {"overall_score": 60},
            }],
            evolution_best_candidate_version=1,
        )
        result = _run(state)
        # best composite = 60 < 80 → 仍更新 best
        assert result["evolution_best_candidate_version"] == 2
