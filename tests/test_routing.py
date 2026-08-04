"""Tests for ModelRouter.is_reasoning 声明（env 驱动，不写死模型名）。"""

from novel_agent.routing import ModelRouter, TaskClass


def test_budget_is_reasoning_from_env(monkeypatch):
    monkeypatch.setenv("BUDGET_IS_REASONING", "true")
    assert ModelRouter().resolve(TaskClass.STRUCTURAL).is_reasoning is True


def test_quality_is_reasoning_from_env(monkeypatch):
    monkeypatch.setenv("QUALITY_IS_REASONING", "1")
    assert ModelRouter().resolve(TaskClass.CREATIVE).is_reasoning is True


def test_default_not_reasoning(monkeypatch):
    monkeypatch.delenv("BUDGET_IS_REASONING", raising=False)
    monkeypatch.delenv("QUALITY_IS_REASONING", raising=False)
    assert ModelRouter().resolve(TaskClass.REVIEW).is_reasoning is False
    assert ModelRouter().resolve(TaskClass.CREATIVE).is_reasoning is False
