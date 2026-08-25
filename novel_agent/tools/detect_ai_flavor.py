"""detect_ai_flavor tool — rule-based AI writing pattern detection."""

from pydantic import BaseModel, Field

from novel_agent.style.analyzer import StyleAnalyzer
from novel_agent.tools.base import BaseTool, ToolResult


class DetectAiFlavorInput(BaseModel):
    """Input schema for detect_ai_flavor tool."""
    text: str = Field(description="The text to analyze for AI writing patterns")


class DetectAiFlavorTool(BaseTool):
    name = "detect_ai_flavor"
    description = (
        "Analyze text for AI writing patterns "
        "(banned phrases, clichés, structural uniformity)."
    )

    @property
    def input_schema(self) -> type[DetectAiFlavorInput]:
        return DetectAiFlavorInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = DetectAiFlavorInput(**kwargs)
        report = StyleAnalyzer().legacy_report(inp.text)
        return ToolResult(success=True, data=report)
