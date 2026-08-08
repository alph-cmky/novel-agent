"""Tests for ContinuityAgent tool-hint gating — 无工具时不提示 check_continuity。

project_id 为空时 CheckContinuityTool 未注册，system_prompt / audit() 的 user prompt
仍硬编码「使用/先用 check_continuity 工具」会让模型输出 <check_continuity> 文本标签
并产出空/坏报告（与 Writer 同款 bug）。
"""

import asyncio
from unittest.mock import AsyncMock, patch

from novel_agent.agents.continuity import ContinuityAgent


class _FakeTool:
    name = "check_continuity"


class TestContinuityToolHint:
    @staticmethod
    def _capture_messages(auditor: ContinuityAgent) -> tuple[str, str]:
        with patch.object(
            auditor, "run_with_tools", new=AsyncMock(return_value=("{}", None))
        ) as mocked:
            asyncio.run(auditor.audit(chapter_number=1, draft_content="正文"))
        system = mocked.call_args.args[0][0]["content"]
        user = mocked.call_args.args[0][1]["content"]
        return system, user

    def test_no_tool_omits_tool_hints(self):
        system, user = self._capture_messages(ContinuityAgent())
        assert "check_continuity" not in system
        assert "check_continuity" not in user
        assert "直接逐项比对" in user

    def test_tool_registered_includes_tool_hints(self):
        auditor = ContinuityAgent()
        auditor.register_tool(_FakeTool())
        system, user = self._capture_messages(auditor)
        assert "check_continuity" in system
        assert "check_continuity" in user
        assert "直接逐项比对" not in user
