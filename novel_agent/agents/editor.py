"""Editor Agent — reviews chapter quality and detects AI writing patterns."""

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.schema.parser import parse_json_response
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

    async def review(self, chapter_number: int, draft_content: str) -> tuple[dict, TraceStep]:
        """Review a chapter draft.

        Returns (report_dict, trace).
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请审查第{chapter_number}章的正文。\n\n"
                    f"## 正文\n{draft_content}\n\n"
                    f"先用 detect_ai_flavor 工具扫描正文，然后给出完整的审查报告。"
                    f"只输出JSON，不要其他内容。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages, max_rounds=2, action=f"review_chapter_{chapter_number}"
        )

        report = parse_json_response(content, defaults={
            "overall_score": 0,
            "dimensions": {},
            "issues": [],
            "highlights": [],
            "verdict": "manual_review",
        })
        return report, trace
