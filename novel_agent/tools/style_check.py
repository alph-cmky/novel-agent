"""analyze_style tool — deterministic style analysis exposed as an agent tool.

Wraps StyleAnalyzer (0 LLM). The Writer agent calls this to self-check
AI flavor before/after humanize_passage, without a separate graph node.
"""

from pydantic import BaseModel, Field

from novel_agent.style.analyzer import StyleAnalyzer
from novel_agent.tools.base import BaseTool, ToolResult


class StyleCheckInput(BaseModel):
    """Input schema for analyze_style tool."""

    text: str = Field(description="要分析的章节正文")


class StyleCheckTool(BaseTool):
    name = "analyze_style"
    description = (
        "确定性分析文本的AI味、段落结构、句子节奏。"
        "返回 ai_flavor_score (0-100)、issues 列表、style_gate。"
        "零 LLM 调用，纯文本测量。"
    )

    @property
    def input_schema(self) -> type[StyleCheckInput]:
        return StyleCheckInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = StyleCheckInput(**kwargs)
        report = StyleAnalyzer().analyze(inp.text)

        issues_summary = [
            {
                "type": i.type,
                "phrase": getattr(i, "phrase", None) or getattr(i, "pattern", ""),
                "count": i.count,
            }
            for i in report.issues
        ]

        if report.ai_flavor_score >= 85 and not report.issues:
            instruction = (
                f"AI味评分 {report.ai_flavor_score}/100，未发现问题。"
                "原文人味足够，跳过 humanize_passage，直接用作终稿。"
            )
        elif report.issues:
            patterns = "、".join(
                (getattr(i, "phrase", None) or getattr(i, "pattern", "")) for i in report.issues[:5]
            )
            instruction = (
                f"AI味评分 {report.ai_flavor_score}/100，发现 {len(report.issues)} 项问题"
                f"（{patterns}）。建议调用 humanize_passage 去除AI味。"
            )
        else:
            instruction = f"AI味评分 {report.ai_flavor_score}/100，无明显禁用词但结构可能偏AI。"

        return ToolResult(
            success=True,
            data={
                "ai_flavor_score": report.ai_flavor_score,
                "style_gate": report.style_gate,
                "issues": issues_summary,
                "issue_count": len(report.issues),
                "paragraph_structure_score": report.paragraph_structure_score,
                "sentence_rhythm_score": report.sentence_rhythm_score,
                "instruction": instruction,
            },
        )
