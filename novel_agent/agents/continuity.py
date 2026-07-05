"""Continuity Agent — cross-chapter consistency audit."""

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.parser import parse_json_response
from novel_agent.tools.continuity import CheckContinuityTool

CONTINUITY_SYSTEM_PROMPT = """你是一个长篇小说设定审计员，专门检查章节间的一致性。

## 审计维度

### 1. 角色一致性
- 外貌（眼睛颜色、身高体型、特征标记）是否前后一致？
- 性格标签是否一致？行为是否符合已建立的性格？
- 能力体系是否被破坏？（主角突然会了不该会的技能）
- 人际关系是否矛盾？（前文说A和B第一次见，本章说他们认识很久）

### 2. 时间线一致性
- 事件顺序是否有矛盾？
- 时间流逝描述是否合理？
- 角色年龄是否匹配？

### 3. 世界观一致性
- 规则体系是否被违反？（前文说魔法需念咒，本章默发）
- 势力关系是否前后矛盾？
- 物品状态是否一致？（前文说宝剑已断，本章又在用）

## 严重程度定义
- **critical**: 核心设定被破坏，不修改会崩世界观
- **major**: 明显矛盾，读者可能注意到
- **minor**: 小瑕疵（如颜色记错、数字不一致）

## 叙事模式感知

根据项目的 narrative_mode 调整审计策略：

- **linear（线性）**：标准3维度审计
- **unit_arc（单元剧）**：角色状态在单元之间可"重置"，不做跨单元状态一致性检查
- **multi_perspective（多视角）**：逐 POV 线审计，不同 POV 线间允许信息不一致，但同一线内部必须一致
- **ensemble（群像）**：多线并行，每个角色线独立审计
- **flashback / non_linear**：不做"时间推进方向"检查，但检查事件因果一致性

## 工具
使用 check_continuity 工具检索前文的角色/事件/世界观信息进行比对。

## 输出格式

```json
{
  "overall_score": 0-100,
  "inconsistencies": [
    {
      "type": "character|timeline|worldbuilding",
      "severity": "critical|major|minor",
      "location_current": "本章描述内容",
      "location_previous": "前文矛盾内容",
      "description": "具体冲突描述",
      "fix_suggestion": "修复建议"
    }
  ],
  "verdict": "pass|rewrite|minor_fix"
}
```
"""


class ContinuityAgent(BaseAgent):
    name = "continuity"

    def __init__(
        self,
        config: AgentConfig | None = None,
        chapter_store: ChapterStore | None = None,
        project_id: str = "",
    ):
        super().__init__(config)
        self._project_id = project_id
        if chapter_store and project_id:
            self.register_tool(CheckContinuityTool(chapter_store, project_id))

    @property
    def system_prompt(self) -> str:
        return CONTINUITY_SYSTEM_PROMPT

    async def audit(
        self, chapter_number: int, draft_content: str,
        narrative_mode: str | None = None,
    ) -> tuple[dict, TraceStep]:
        """Audit chapter for continuity issues.

        Returns (audit_report, trace).
        """
        mode_hint = ""
        if narrative_mode:
            mode_hint = f"\n当前叙事模式：{narrative_mode}。请根据叙事模式调整审计策略。\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请审计第{chapter_number}章的设定一致性。\n"
                    f"{mode_hint}\n"
                    f"## 本章正文\n{draft_content[:4000]}\n\n"
                    f"先用 check_continuity 工具检索前文的相关设定，"
                    f"然后逐项比对给出审计报告。只输出JSON，不要其他内容。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages, max_rounds=2, action=f"audit_chapter_{chapter_number}"
        )

        report = parse_json_response(content, defaults={
            "overall_score": 0,
            "inconsistencies": [],
            "verdict": "manual_review",
        })
        return report, trace
