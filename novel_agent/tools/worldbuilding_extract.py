"""worldbuilding_extract tool — wraps WorldbuildingAgent as a callable tool.

The Writer agent calls this when chapter content is stable to extract
entities, conflicts, and foreshadowings. Results are stored in a shared
context dict for the agent_loop runner to persist.
"""

from pydantic import BaseModel, Field

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.worldbuilding import WorldbuildingAgent
from novel_agent.tools.base import BaseTool, ToolResult


class WorldbuildingExtractInput(BaseModel):
    """Input schema for worldbuilding_extract tool."""

    chapter_content: str = Field(description="要提取设定信息的章节正文")
    chapter_number: int = Field(description="当前章号")


class WorldbuildingExtractTool(BaseTool):
    name = "worldbuilding_extract"
    description = (
        "从章节正文中提取实体、冲突、伏笔等设定信息。内容稳定后调用一次即可。返回提取报告。"
    )

    def __init__(
        self,
        config: AgentConfig | None = None,
        existing_entities: list[dict] | None = None,
        existing_foreshadowings: list[dict] | None = None,
        narrative_mode: str | None = None,
        shared: dict | None = None,
    ):
        self._agent = WorldbuildingAgent(
            config,
            existing_entities=existing_entities,
            existing_foreshadowings=existing_foreshadowings,
        )
        self._narrative_mode = narrative_mode
        self._shared = shared or {}

    @property
    def input_schema(self) -> type[WorldbuildingExtractInput]:
        return WorldbuildingExtractInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = WorldbuildingExtractInput(**kwargs)

        report, _ = await self._agent.extract(
            chapter_number=inp.chapter_number,
            draft_content=inp.chapter_content,
            narrative_mode=self._narrative_mode,
        )

        self._shared["worldbuilding_report"] = report

        entities = report.get("entities") or []
        conflicts = report.get("conflicts") or []
        foreshadowings = report.get("foreshadowings") or []

        return ToolResult(
            success=True,
            data={
                "entity_count": len(entities),
                "conflict_count": len(conflicts),
                "foreshadowing_count": len(foreshadowings),
                "instruction": (
                    f"已提取 {len(entities)} 个实体、{len(conflicts)} 个冲突、"
                    f"{len(foreshadowings)} 个伏笔。设定已记录。"
                ),
            },
        )
