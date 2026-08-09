"""Continuity Agent — cross-chapter consistency audit."""

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.parser import parse_json_response
from novel_agent.schema.validator import OutputValidator
from novel_agent.tools.continuity import CheckContinuityTool

CONTINUITY_TOOL_SECTION = """## 工具
使用 check_continuity 工具检索前文的角色/事件/世界观信息进行比对。

"""

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

{TOOL_SECTION}## 输出格式

```json
{
  "overall_score": 0-100,
  "inconsistencies": [
    {
      "category": "character|timeline|worldbuilding",
      "severity": "critical|major|minor",
      "current": "本章描述内容",
      "previous": "前文矛盾内容",
      "description": "具体冲突描述",
      "suggestion": "修复建议"
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
        tool = CONTINUITY_TOOL_SECTION if self._tools else ""
        return CONTINUITY_SYSTEM_PROMPT.replace("{TOOL_SECTION}", tool)

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

        # 仅当 check_continuity 工具已注册（project_id 非空）才提示使用工具；
        # 否则模型无法调用工具，会输出 <check_continuity> 文本标签并产出空/坏报告。
        tool_hint = (
            "先用 check_continuity 工具检索前文的相关设定，然后逐项比对给出审计报告。"
            if self._tools
            else "直接逐项比对正文与前文设定，给出审计报告。"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请审计第{chapter_number}章的设定一致性。\n"
                    f"{mode_hint}\n"
                    f"## 本章正文\n{draft_content[:4000]}\n\n"
                    f"{tool_hint}只输出JSON，不要其他内容。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages, max_rounds=2, action=f"audit_chapter_{chapter_number}"
        )

        defaults = {
            "overall_score": 0,
            "inconsistencies": [],
            "verdict": "manual_review",
        }

        # 两类「拿不到有效审计报告」的情况都标记 unavailable，避免兜底
        # overall_score=0 被进化层误读为「最差版本」触发 regressed：
        #   1. 空输出（reasoning 模型偶发空 content）
        #   2. 非空但解析失败（JSON 语法错误/截断）—— parse_json_response 失败时
        #      会带 raw_output 哨兵字段
        if not (content or "").strip():
            return {"unavailable": True, **defaults}, trace

        raw = parse_json_response(content, defaults=defaults)
        if "raw_output" in raw:
            return {"unavailable": True, **defaults}, trace

        report = OutputValidator.validate("continuity", raw).to_dict()
        return report, trace
