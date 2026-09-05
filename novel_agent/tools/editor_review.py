"""editor_review tool — wraps EditorAgent as a callable tool for the Writer loop.

The Writer agent calls this to get 8-dimension literary feedback without a
fixed graph node. Results are also stored in a shared context dict so the
agent_loop runner can extract the final report into NovelState.
"""

from pydantic import BaseModel, Field

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.editor import EditorAgent
from novel_agent.style.analyzer import StyleAnalyzer
from novel_agent.tools.base import BaseTool, ToolResult


class EditorReviewInput(BaseModel):
    """Input schema for editor_review tool."""

    chapter_content: str = Field(description="要审查的章节正文")
    chapter_number: int = Field(description="当前章号")


class EditorReviewTool(BaseTool):
    name = "editor_review"
    description = (
        "调用金牌主审编辑审查章节，返回8维度评分（连贯性/文笔/AI味/对话/情节/"
        "大纲还原/创意/可操控性）、issues 列表和 verdict（pass/minor_fix/rewrite）。"
    )

    def __init__(
        self,
        config: AgentConfig | None = None,
        narrative_mode: str | None = None,
        context_packet: dict | None = None,
        shared: dict | None = None,
    ):
        self._agent = EditorAgent(config)
        self._narrative_mode = narrative_mode
        self._context_packet = context_packet
        self._shared = shared or {}

    @property
    def input_schema(self) -> type[EditorReviewInput]:
        return EditorReviewInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = EditorReviewInput(**kwargs)

        style_report = StyleAnalyzer().analyze(inp.chapter_content).model_dump()

        report, _ = await self._agent.review(
            chapter_number=inp.chapter_number,
            draft_content=inp.chapter_content,
            narrative_mode=self._narrative_mode,
            style_report=style_report,
            context_packet=self._context_packet,
        )

        self._shared["editor_report"] = report
        self._shared["style_report"] = style_report

        verdict = report.get("verdict", "manual_review")
        overall = report.get("overall_score", 0)
        issues = report.get("issues") or []
        issue_count = len(issues)

        if verdict == "pass":
            instruction = f"编辑评分 {overall}/100，verdict=pass。章节质量达标。"
        elif issue_count:
            top_issues = "; ".join(
                f"{i.get('dimension', '?')}: {i.get('description', '')[:60]}" for i in issues[:3]
            )
            instruction = (
                f"编辑评分 {overall}/100，verdict={verdict}。"
                f"主要问题：{top_issues}。请据此修订正文。"
            )
        else:
            instruction = f"编辑评分 {overall}/100，verdict={verdict}。"

        return ToolResult(
            success=True,
            data={
                "overall_score": overall,
                "verdict": verdict,
                "issues": issues,
                "issue_count": issue_count,
                "instruction": instruction,
            },
        )
