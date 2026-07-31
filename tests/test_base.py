"""Tests for BaseAgent reasoning-model handling — reasoning_effort injection."""

from novel_agent.agents.base import (
    AgentConfig,
    _build_chat_model,
    _is_reasoning_model,
)


class TestIsReasoningModel:
    def test_stepfun_base_url(self):
        assert _is_reasoning_model("step-3.7-flash", "https://api.stepfun.com/step_plan/v1")

    def test_step_model_name(self):
        assert _is_reasoning_model("step-3.7-flash", "")

    def test_plain_openai_model_is_not_reasoning(self):
        assert not _is_reasoning_model("gpt-4o", "https://api.openai.com/v1")

    def test_empty_inputs_safe(self):
        assert not _is_reasoning_model("", "")


class TestBuildChatModel:
    @staticmethod
    def _config(model: str, base_url: str) -> AgentConfig:
        return AgentConfig(model=model, api_key="test-key", base_url=base_url)

    def test_reasoning_model_injects_low_effort(self):
        model = _build_chat_model(
            self._config("step-3.7-flash", "https://api.stepfun.com/step_plan/v1")
        )
        assert model.reasoning_effort == "low"

    def test_non_reasoning_model_no_effort(self):
        model = _build_chat_model(
            self._config("gpt-4o", "https://api.openai.com/v1")
        )
        assert model.reasoning_effort is None
