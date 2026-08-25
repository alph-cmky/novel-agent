"""search_context tool — semantic search across chapters."""

from typing import Literal

from pydantic import BaseModel, Field

from novel_agent.memory.embeddings import ChapterStore
from novel_agent.tools.base import BaseTool, ToolResult


class SearchContextInput(BaseModel):
    """Input schema for search_context tool."""

    query: str = Field(description="Search query or description")
    scope: Literal["characters", "events", "locations", "foreshadowings", "all"] = "all"
    top_k: int = Field(default=5, ge=1, le=20)
    chapter_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional chapter range as (start, end)",
    )


class SearchContextTool(BaseTool):
    name = "search_context"
    description = "Semantically search across all written chapters to find relevant context."

    def __init__(self, chapter_store: ChapterStore, project_id: str):
        self._store = chapter_store
        self._project_id = project_id

    @property
    def input_schema(self) -> type[SearchContextInput]:
        return SearchContextInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = SearchContextInput(**kwargs)
        results = self._store.search(
            project_id=self._project_id,
            query=inp.query,
            top_k=inp.top_k,
            chapter_range=inp.chapter_range,
        )
        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_found": len(results),
                "search_scope": inp.scope,
            },
        )
