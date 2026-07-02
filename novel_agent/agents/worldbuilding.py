"""Worldbuilding Agent — entity extraction, conflict detection, knowledge graph.

Reads chapter output, extracts new entities (characters, locations, rules),
compares against existing worldbuilding database, and flags conflicts.
"""

import json

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.schema.parser import parse_json_response

WORLDBUILDING_SYSTEM_PROMPT = """你是一个小说世界观管理员，负责从章节内容中提取和管理设定。

## 你的职责

### 1. 提取新设定
从章节中识别并结构化以下类型的设定：

- **角色 (character)**: 新出场角色的姓名、外貌、性格、能力、人际关系
- **地点 (location)**: 新场景的名称、特征、所属势力
- **势力 (faction)**: 组织、门派、家族及其关系和规则
- **规则 (rule)**: 世界观规则体系（魔法规则、修炼体系、社会制度等）
- **物品 (item)**: 重要道具、神器、信物及其特性
- **事件 (event)**: 历史事件、关键转折

### 2. 冲突检测
将新提取的设定与已知设定比对，检测潜在冲突：
- 同名角色描述不一致
- 新规则与旧规则矛盾
- 时间线冲突

### 3. 输出格式

```json
{
  "new_entities": [
    {
      "entity_type": "character|location|faction|rule|item|event",
      "name": "实体名称",
      "properties": {"key": "value"},
      "first_appearance_chapter": 3,
      "relationships": [
        {"target": "关联实体名", "relation": "关系描述"}
      ]
    }
  ],
  "conflicts": [
    {
      "existing_entity": "已有实体名",
      "conflict_type": "description_mismatch|rule_violation|timeline",
      "description": "冲突描述",
      "new_info": "本章中的描述",
      "existing_info": "已有设定中的描述",
      "severity": "critical|major|minor"
    }
  ],
  "chapter_events": ["本章发生的关键事件"],
  "updated_entities": ["设定有更新或进展的已有实体"]
}
```

只输出JSON，不要其他内容。
"""


class WorldbuildingAgent(BaseAgent):
    name = "worldbuilding"

    def __init__(
        self,
        config: AgentConfig | None = None,
        existing_entities: list[dict] | None = None,
    ):
        super().__init__(config)
        self._existing_entities = existing_entities or []

    @property
    def system_prompt(self) -> str:
        return WORLDBUILDING_SYSTEM_PROMPT

    async def extract(
        self,
        chapter_number: int,
        draft_content: str,
    ) -> tuple[dict, TraceStep]:
        """Extract worldbuilding entities from a chapter.

        Returns (extraction_report, trace).
        """
        existing_json = json.dumps(
            self._existing_entities, ensure_ascii=False, indent=2
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"从第{chapter_number}章提取新设定，并与已有设定比对。\n\n"
                    f"## 已有设定\n{existing_json}\n\n"
                    f"## 本章正文\n{draft_content[:4000]}\n\n"
                    f"只输出JSON。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"extract_worldbuilding_ch{chapter_number}",
        )

        report = parse_json_response(content, defaults={
            "new_entities": [],
            "conflicts": [],
            "chapter_events": [],
            "updated_entities": [],
        })
        return report, trace

    def set_existing_entities(self, entities: list[dict]):
        """Update the existing entities for comparison."""
        self._existing_entities = entities
