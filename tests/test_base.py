"""Tests for BaseAgent reasoning-model handling — is_reasoning 声明与 build_chat_model."""

import pytest

from novel_agent.agents.base import AgentConfig, build_chat_model


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
