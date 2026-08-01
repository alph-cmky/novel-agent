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
        draft_preview="",
    )


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
