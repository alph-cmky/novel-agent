"""Tests for EvolutionOrchestratorAgent.enrich_plan — LLM 输出规范化。

strategy_override 必须是 dict（writer_node 会 dict.update）。LLM 偶发输出
list/str 导致 ValueError，这里验证 enrich_plan 在源头把非 dict 规范化掉。
"""

import asyncio
import json

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.evolution_orchestrator import EvolutionOrchestratorAgent


def _make_rule_plan() -> dict:
    return {
        "focus_dimensions": ["dialogue"],
        "primary_instruction": "规则层核心指令",
        "secondary_instructions": [],
        "constraints": {"preserve": [], "avoid": [], "strategy_override": {}},
    }


async def _enrich(result: dict) -> dict:
    agent = EvolutionOrchestratorAgent(config=AgentConfig(model="test", api_key="test-key"))

    async def fake_run_with_tools(messages, **kwargs):
        return json.dumps(result, ensure_ascii=False), None

    agent.run_with_tools = fake_run_with_tools
    return await agent.enrich_plan(
        current_version=1,
        current_scores={"editor_overall": 80, "continuity_overall": 80, "dimensions": {}},
        delta=None,
        rule_plan=_make_rule_plan(),
        history=[],
        violations=[],
    )


class TestEnrichPlanContext:
    """Phase H: enrichment 上下文仅限 scores / delta / violations / rule_plan。"""

    @staticmethod
    def _capture_user_prompt(violations=None, delta=None) -> str:
        agent = EvolutionOrchestratorAgent(config=AgentConfig(model="test", api_key="test-key"))
        captured = {}

        async def fake_run_with_tools(messages, **kwargs):
            captured["user"] = messages[1]["content"]
            return '{"primary_instruction": "x"}', None

        agent.run_with_tools = fake_run_with_tools
        asyncio.run(
            agent.enrich_plan(
                current_version=2,
                current_scores={"editor_overall": 80, "continuity_overall": 75, "dimensions": {}},
                delta=delta,
                rule_plan=_make_rule_plan(),
                history=[{"v": 1, "editor": 70, "continuity": 70}],
                violations=violations,
            )
        )
        return captured["user"]

    def test_violations_rendered_in_prompt(self):
        user = self._capture_user_prompt(violations=["style_gate_fail", "length_regression"])
        assert "style_gate_fail" in user
        assert "length_regression" in user
        assert "质量门违规" in user

    def test_no_violations_renders_none(self):
        user = self._capture_user_prompt(violations=[])
        assert "质量门违规\n无" in user

    def test_prompt_has_no_draft_section(self):
        """enrichment prompt 不包含草稿预览段落（Phase H 移除 draft_preview）。"""
        user = self._capture_user_prompt(violations=["style_gate_fail"])
        assert "草稿" not in user
        assert "draft" not in user.lower()


class TestEnrichPlanStrategyOverride:
    def test_list_strategy_override_normalized_to_empty(self):
        result = {
            "primary_instruction": "改进对话",
            "constraints": {
                "preserve": [],
                "avoid": [],
                "strategy_override": ["快节奏", "多对话"],
            },
        }
        plan = asyncio.run(_enrich(result))
        assert plan["constraints"]["strategy_override"] == {}

    def test_str_strategy_override_normalized_to_empty(self):
        result = {
            "primary_instruction": "改进对话",
            "constraints": {
                "preserve": [],
                "avoid": [],
                "strategy_override": "快节奏",
            },
        }
        plan = asyncio.run(_enrich(result))
        assert plan["constraints"]["strategy_override"] == {}

    def test_dict_strategy_override_kept(self):
        result = {
            "primary_instruction": "改进对话",
            "constraints": {
                "preserve": [],
                "avoid": [],
                "strategy_override": {"pacing": "快"},
            },
        }
        plan = asyncio.run(_enrich(result))
        assert plan["constraints"]["strategy_override"] == {"pacing": "快"}
