"""check_continuity tool — cross-chapter consistency audit."""

from pydantic import BaseModel, Field

from novel_agent.memory.embeddings import ChapterStore
from novel_agent.tools.base import BaseTool, ToolResult


class CheckContinuityInput(BaseModel):
    """Input schema for check_continuity tool."""
    chapter_content: str = Field(description="Current chapter content to audit")
    chapter_number: int = Field(description="Current chapter number")
    project_id: str = Field(description="Project ID for context retrieval")


class CheckContinuityTool(BaseTool):
    name = "check_continuity"
    description = (
        "Audit current chapter against previous chapters for character, "
        "timeline, and worldbuilding consistency. Requires project_id and "
        "chapter_number to retrieve context."
    )

    def __init__(self, chapter_store: ChapterStore, project_id: str):
        self._store = chapter_store
        self._project_id = project_id

    @property
    def input_schema(self) -> type[CheckContinuityInput]:
        return CheckContinuityInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = CheckContinuityInput(**kwargs)
        # Retrieve relevant past context for comparison
        char_results = self._store.search(
            project_id=inp.project_id,
            query="角色 性格 外貌 能力 人际关系",
            top_k=5,
            chapter_range=(1, inp.chapter_number - 1) if inp.chapter_number > 1 else None,
        )
        event_results = self._store.search(
            project_id=inp.project_id,
            query="时间线 事件 因果关系",
            top_k=5,
            chapter_range=(1, inp.chapter_number - 1) if inp.chapter_number > 1 else None,
        )
        world_results = self._store.search(
            project_id=inp.project_id,
            query="世界观 规则 设定 势力 魔法",
            top_k=5,
            chapter_range=(1, inp.chapter_number - 1) if inp.chapter_number > 1 else None,
        )

        context = {
            "character_context": [r["content"][:300] for r in char_results],
            "event_context": [r["content"][:300] for r in event_results],
            "world_context": [r["content"][:300] for r in world_results],
        }

        return ToolResult(
            success=True,
            data={
                "retrieved_context": context,
                "note": (
                    "Retrieved past context for continuity comparison. "
                    "The ContinuityAgent will analyze consistency with this data."
                ),
            },
        )
