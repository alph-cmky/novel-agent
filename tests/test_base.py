"""Tests for BaseAgent reasoning-model handling — is_reasoning 声明与 build_chat_model."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novel_agent.agents.base import AgentConfig, BaseAgent, build_chat_model


class TestAgentConfigIsReasoning:
    def test_default_false(self):
        assert AgentConfig(model="gpt-4o", api_key="k").is_reasoning is False

    def test_explicit_true(self):
        config = AgentConfig(model="gpt-4o", api_key="k", is_reasoning=True)
        assert config.is_reasoning is True

    def test_non_bool_raises(self):
        with pytest.raises(ValueError):
            AgentConfig(model="gpt-4o", api_key="k", is_reasoning="yes")


class TestBuildChatModel:
    @staticmethod
    def _config(is_reasoning: bool = False) -> AgentConfig:
        return AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            is_reasoning=is_reasoning,
        )

    def test_reasoning_model_injects_low_effort(self):
        model = build_chat_model(self._config(is_reasoning=True))
        assert model.reasoning_effort == "low"

    def test_reasoning_model_disables_stream_chunk_timeout(self):
        # 推理模型长 thinking 阶段 >120s 不吐 chunk，禁用流式超时避免误判
        model = build_chat_model(self._config(is_reasoning=True))
        assert model.stream_chunk_timeout is None

    def test_non_reasoning_model_no_effort(self):
        model = build_chat_model(self._config(is_reasoning=False))
        assert model.reasoning_effort is None

    def test_non_reasoning_model_keeps_stream_chunk_timeout(self):
        model = build_chat_model(self._config(is_reasoning=False))
        assert model.stream_chunk_timeout is not None


class _StubAgent(BaseAgent):
    name = "stub"

    @property
    def system_prompt(self) -> str:
        return "stub"


class TestCostCounters:
    """Phase I/J: agent 实例级成本计数器。

    model_calls 按实际 API 请求计（含空输出重试），tool_call_counts
    按工具名累计——调用方（writer_node）读取后写入 state 成本观测字段。
    """

    @staticmethod
    def _agent() -> _StubAgent:
        return _StubAgent(AgentConfig(model="gpt-4o", api_key="k"))

    def test_initial_counters_zero(self):
        agent = self._agent()
        assert agent.model_calls == 0
        assert agent.tool_call_counts == {}

    def test_run_with_tools_counts_model_and_tool_calls(self):
        """run_with_tools 循环：model 请求 2 次（工具轮+正文轮），工具 1 次。"""
        from novel_agent.tools.base import BaseTool, ToolInput

        class _SearchTool(BaseTool):
            name = "search_context"
            description = "stub"

            @property
            def input_schema(self):
                return ToolInput

            async def execute(self, **kwargs):
                from novel_agent.tools.base import ToolResult

                return ToolResult(success=True, data={})

        agent = self._agent()
        agent.register_tool(_SearchTool())

        tool_round = MagicMock(
            content="",
            tool_calls=[{"id": "t1", "name": "search_context", "args": {"query": "x"}}],
        )
        final_round = MagicMock(content="正文", tool_calls=[])
        # side_effect 必须挂在 ainvoke 上：父级 AsyncMock 的 side_effect
        # 不会传播给子 mock，否则 await 拿到的是默认 AsyncMock。
        fake_bound = MagicMock()
        fake_bound.ainvoke = AsyncMock(side_effect=[tool_round, final_round])
        fake_model = MagicMock()
        fake_model.bind_tools.return_value = fake_bound

        with patch("novel_agent.agents.base.build_chat_model", return_value=fake_model):
            content, _ = asyncio.run(
                agent.run_with_tools(
                    [{"role": "user", "content": "x"}],
                    max_rounds=3,
                )
            )
        assert content == "正文"
        assert agent.model_calls == 2
        assert agent.tool_call_counts == {"search_context": 1}
