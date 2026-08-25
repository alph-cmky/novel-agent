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

## 核心审计维度（对齐严苛事实一致性标准）

### 1. 角色与实体一致性（Character & Entity Consistency）
- **姓名与身份**：绝不允许出现受害者/角色前后改名（如前文叫周婉，本章误写为林晚）。
- **人际与血缘关系**：严查生父/养父/师徒等关键人设关系是否混淆（如养女误写为生父教导）。
- **能力与状态**：技能等级、伤病状态、持有道具是否与前文事实完全吻合。

### 2. 时间线与数值逻辑（Timeline & Numeric Logic）
- **时间跨度与时长**：严查天数、年份、间隔时间（如前文写三天前失踪，本章账本写五天前，属于重大矛盾）。
- **事件先后因果**：前置事件是否发生？死亡角色是否复活？

### 3. 世界观与场景实体（Worldbuilding & Physical Rules）
- **材质与物理属性**：城墙材质（黑曜石 vs 青石）、法宝规则、门派试炼场数（三场 vs 两场）等硬性设定是否矛盾。
- **势力与地域规则**：地理位置、宗门禁地与势力阵营前后是否自洽。

## 严重程度定义与打分标准
- **critical (-30~50分)**：核心设定被破坏、主要角色改名/身世矛盾、核心时间线崩溃。
- **major (-15~30分)**：局部规则冲突、次要时间矛盾、读者能明显察觉的不一致。
- **minor (-5~10分)**：用词瑕疵、无关紧要的微小数值出入。
- **若存在 1 处 critical 矛盾，overall_score 严禁超过 60 分！**

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
        self,
        chapter_number: int,
        draft_content: str,
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
                    f"## 本章正文\n{draft_content}\n\n"
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

        # 若首轮返回空或解析失败，针对 reasoning 模型执行一次简化重试（直接 prompt 约束）
        for attempt in range(2):
            if attempt > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "请务必只输出合法的 JSON 格式审计结果，包含 overall_score, inconsistencies, verdict。",
                    }
                )
                content, trace = await self.run_with_tools(
                    messages, max_rounds=1, action=f"audit_chapter_{chapter_number}_retry"
                )

            if not (content or "").strip():
                continue

            raw = parse_json_response(content, defaults=defaults)
            if "raw_output" not in raw:
                report = OutputValidator.validate("continuity", raw).to_dict()
                return report, trace

        return {"unavailable": True, **defaults}, trace
