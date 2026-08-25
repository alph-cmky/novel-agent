"""Worldbuilding Agent — entity extraction, conflict detection, foreshadowing lifecycle.

Reads chapter output, extracts new entities (characters, locations, rules),
compares against existing worldbuilding database, flags conflicts, and manages
foreshadowing lifecycle (plant/resolve/advance).
"""

import json

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.schema.validator import parse_validated

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

### 3. 伏笔生命周期管理
从本章内容中识别伏笔（foreshadowing），并与已有伏笔比对：

**新伏笔识别标准：**
- 角色提及未来计划/约定（"下次见面时..."、"三个月后..."）
- 物品/能力出现但未解释（神秘道具、未知力量）
- 信息不对称（某人知道但读者/其他人不知道）
- 预言或暗示（"这个选择将改变一切"）
- 未完成的对话/行动

**已有伏笔状态判断：**
- 被解决了 → 放入 resolved_foreshadowings
- 有进展但未解决 → 放入 foreshadowings（只保留有实质进展的）
- 无进展 → 不需要在输出中提及

### 4. 输出格式

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
  "updated_entities": [
    {
      "entity_type": "character|location|faction|rule|item|event",
      "name": "已有实体名称",
      "properties": {"发生变化的属性": "新值"}
    }
  ],
  "foreshadowings": [
    {
      "description": "伏笔描述",
      "planted_chapter": 3,
      "expected_resolve_chapter": null,
      "risk_level": "high|medium|low",
      "action_needed": "后续章节应如何回应此伏笔",
      "reader_knows": true,
      "characters_aware": ["角色名"],
      "characters_unaware": ["角色名"]
    }
  ],
  "resolved_foreshadowings": [
    {
      "description": "已解决伏笔的原文（需与已有伏笔匹配）",
      "resolved_chapter": 3
    }
  ]
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
        existing_foreshadowings: list[dict] | None = None,
    ):
        super().__init__(config)
        self._existing_entities = existing_entities or []
        self._existing_foreshadowings = existing_foreshadowings or []

    @property
    def system_prompt(self) -> str:
        return WORLDBUILDING_SYSTEM_PROMPT

    async def extract(
        self,
        chapter_number: int,
        draft_content: str,
        narrative_mode: str | None = None,
    ) -> tuple[dict, TraceStep]:
        """Extract worldbuilding entities, conflicts, and foreshadowings from a chapter.

        Returns (extraction_report, trace).
        """
        existing_json = json.dumps(self._existing_entities, ensure_ascii=False, indent=2)

        # Build existing foreshadowings context
        fs_context = ""
        if self._existing_foreshadowings:
            open_fs = [
                f for f in self._existing_foreshadowings if f.get("status") in ("open", "planted")
            ]
            if open_fs:
                items = []
                for f in open_fs:
                    items.append(
                        f"  - [{f.get('risk_level', '?')}] "
                        f"第{f.get('planted_chapter', '?')}章: {f.get('description', '')}"
                    )
                fs_context = "## 已有伏笔（待解决/有进展）\n" + "\n".join(items) + "\n"

        # Mode-specific instruction for POV tagging
        mode_hint = ""
        if narrative_mode in ("multi_perspective", "ensemble"):
            mode_hint = (
                "\n当前为多视角/群像模式。请在实体的 properties 中增加 "
                '"pov_character" 字段，标注该实体信息来自哪个 POV 角色的视角。\n'
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"从第{chapter_number}章提取新设定、冲突和伏笔。\n"
                    f"{mode_hint}\n"
                    f"## 已有设定\n{existing_json}\n\n"
                    f"{fs_context}"
                    f"## 本章正文\n{draft_content}\n\n"
                    f"只输出JSON。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"extract_worldbuilding_ch{chapter_number}",
        )

        report = parse_validated(
            "worldbuilding",
            content,
            defaults={
                "new_entities": [],
                "conflicts": [],
                "chapter_events": [],
                "updated_entities": [],
                "foreshadowings": [],
                "resolved_foreshadowings": [],
            },
        )
        return report, trace

    def set_existing_entities(self, entities: list[dict]):
        """Update the existing entities for comparison."""
        self._existing_entities = entities
