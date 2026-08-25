"""Base tool with MCP-compatible schema definition."""

from typing import Any

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base class for tool input schemas."""


class ToolResult(BaseModel):
    """Standard tool result wrapper."""

    success: bool = True
    data: Any = None
    error: str | None = None


class BaseTool:
    """Tool base class with MCP-compatible schema pattern.

    Each tool defines:
    - name: unique tool identifier
    - description: what the tool does (for LLM function calling)
    - input_schema: Pydantic model for input validation
    """

    name: str = ""
    description: str = ""

    def get_schema(self) -> dict:
        """Generate OpenAI-compatible function calling schema."""
        schema = self.input_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool. Subclasses override this."""
        raise NotImplementedError

    @property
    def input_schema(self) -> type[BaseModel]:
        raise NotImplementedError
