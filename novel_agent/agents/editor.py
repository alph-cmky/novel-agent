"""Editor Agent — reviews chapter quality and detects AI writing patterns."""

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.schema.validator import parse_validated
from novel_agent.tools.style import DetectAiFlavorTool

EDITOR_SYSTEM_PROMPT = """你是一个资深网文编辑，擅长审稿和识别AI写作痕迹。

## 审查维度

1. **节奏**：章节是否有起伏？读起来是否拖沓？
2. **AI腔**：有没有重复句式、禁用词、情感标签化描写？
3. **对话**：对话是否自然？是否符合角色性格？
4. **逻辑**：情节前后是否自洽？
5. **文笔**：是否口语化？是否有感官细节？

## 审查规则

- 连续三个长度相近的句子 → 扣分
- 段落以简洁单行结尾 → 加分
- 结尾总结升华 → 严重扣分（网文结尾必须有钩子）
- "此外""然而""值得注意的是" → 扣分
- 情感标签堆砌（"他感到愤怒、悲伤、绝望"）→ 扣分
- 过度使用"——"破折号 → 叙事中扣分，对白中可接受

## 叙事模式感知

根据项目的 narrative_mode 调整评分标准：

- **linear（线性）**：标准5维度评分
- **unit_arc（单元剧）**：降低"主线推进"权重。单元内部剧情自洽即可，单元结尾章恢复正常主线要求
- **multi_perspective（多视角）**：不做 POV 一致性惩罚。不同 POV 的信息不对称是叙事手法而非缺陷
- **ensemble（群像）**：关注主要角色出场平衡，连续3章未提及某主要角色时提示但不扣分
- 在审查时请注意当前叙事模式，据此调整 verdict 判定

## 工具

使用 detect_ai_flavor 工具对正文进行规则扫描。

## 输出格式

审查完成后，必须输出JSON格式的审查报告：

```json
{
  "overall_score": 0-100,
  "dimensions": {
    "rhythm": 0-100,
    "ai_flavor": 0-100,
    "dialogue": 0-100,
    "logic": 0-100,
    "writing": 0-100
  },
  "issues": [
    {"dimension": "...", "severity": "critical|major|minor",
     "description": "...", "suggestion": "..."}
  ],
  "highlights": ["写得好的地方"],
  "verdict": "pass|minor_fix|rewrite"
}
```
"""


class EditorAgent(BaseAgent):
    name = "editor"

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self.register_tool(DetectAiFlavorTool())

    @property
    def system_prompt(self) -> str:
        return EDITOR_SYSTEM_PROMPT

    async def review(
        self, chapter_number: int, draft_content: str,
        narrative_mode: str | None = None,
    ) -> tuple[dict, TraceStep]:
        """Review a chapter draft.

        Returns (report_dict, trace).
        """
        mode_hint = ""
        if narrative_mode:
            mode_hint = f"\n当前叙事模式：{narrative_mode}。请根据叙事模式调整评分标准。\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请审查第{chapter_number}章的正文。\n"
                    f"{mode_hint}\n"
                    f"## 正文\n{draft_content}\n\n"
                    f"先用 detect_ai_flavor 工具扫描正文，然后给出完整的审查报告。"
                    f"只输出JSON，不要其他内容。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages, max_rounds=2, action=f"review_chapter_{chapter_number}"
        )

        report = parse_validated("editor", content, defaults={
            "overall_score": 0,
            "dimensions": {},
            "issues": [],
            "highlights": [],
            "verdict": "manual_review",
        })
        return report, trace
