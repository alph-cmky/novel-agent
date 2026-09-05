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
        instruction = (
            "无匹配。不要说明检索结果，依据已有上下文继续写章节正文。"
            if not results
            else "检索内容仅供核对。不要在章节正文中提及检索、工具或信息是否找到。"
        )
        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_found": len(results),
                "search_scope": inp.scope,
                "instruction": instruction,
            },
        )
