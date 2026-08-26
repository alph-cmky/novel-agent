"""Editor Agent — reviews chapter quality using provided style evidence.

The Editor no longer calls a detect_ai_flavor tool. The deterministic
StyleAnalyzer runs in the graph node and its StyleReport is passed as
context. The Editor focuses on literary judgment: characters, logic,
rhythm, dialogue, semantic AI flavor.
"""

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.schema.parser import parse_json_response
from novel_agent.schema.validator import OutputValidator

EDITOR_SYSTEM_PROMPT = """你是一个极其严苛的网文金牌主审编辑，负责把控正文质量，绝不给面子分，必须依据扣分阶梯真实客观打分。

## 审查维度与满分基准（各维度满分100，初始100起扣）

1. **节奏（rhythm）**：起伏是否抓人？冲突是否密集？有无拖沓注水？
2. **AI腔（ai_flavor）**：是否有公文套话、陈词滥调、情感标签堆砌、长短句单一？
3. **对话（dialogue）**：对白是否有潜台词与冲突？是否生硬如播音腔？
4. **逻辑（logic）**：情节推演是否自洽？有无降智、吃书或跳步？
5. **文笔（writing）**：是否动作化展开（Show, Don't Tell）？是否有感官细节？

## 🚨 严苛扣分阶梯（必须严格执行扣分）

- **字数/篇幅严重不足（硬红线）**：若正文过短（如明显低于2000字或只输出大纲片段），`rhythm` 和 `writing` 直接扣 30-50 分！
- **结尾总结升华/说教**：章节结尾出现"这不仅是...更是..."或道理总结，`ai_flavor` 扣 25 分，`rhythm` 扣 20 分。
- **出现公文禁用词**（此外/不仅如此/更重要的是/至关重要等）：每出现一处，`ai_flavor` 扣 10 分。
- **抽象告知而非展示**（"他感到极度愤怒"）：每出现一处，`writing` 扣 5-10 分。
- **无营养对白/千人一面**：角色缺乏性格辨识度，`dialogue` 扣 15-30 分。
- **逻辑断层/情节唐突**：`logic` 扣 20-40 分。

## 判决标准 (verdict)
- **pass**：各维度得分均 >= 85，且无严重逻辑漏洞。
- **minor_fix**：存在局部小瑕疵或个别 AI 词，得分在 70-84 之间。
- **rewrite**：总分 < 70，或存在严重篇幅缩水、重大逻辑崩溃。

## 风格证据
审查前，系统会提供确定性风格分析报告（StyleReport），包含：
- 段落结构指标（碎片化评分、连续短段、单句段比例等）
- AI 味证据（禁用词、陈词滥调、句式问题）
- 句子节奏、对话占比、结尾分析

这些是客观数据，作为评分参考。你的文学判断优先于数据指标——
数据发现的结构异常（如连续短段过多）应体现在对应维度扣分中。

## 输出格式
审查完成后，输出结构化 JSON 报告：
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
    {"dimension": "rhythm|ai_flavor|dialogue|logic|writing", "severity": "critical|major|minor",
     "description": "具体扣分原因", "suggestion": "修改建议"}
  ],
  "highlights": ["写得好的亮点"],
  "verdict": "pass|minor_fix|rewrite"
}
```"""


class EditorAgent(BaseAgent):
    name = "editor"

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)

    @property
    def system_prompt(self) -> str:
        return EDITOR_SYSTEM_PROMPT

    async def review(
        self,
        chapter_number: int,
        draft_content: str,
        narrative_mode: str | None = None,
        style_report: dict | None = None,
        context_packet: dict | None = None,
    ) -> tuple[dict, TraceStep]:
        """Review a chapter draft.

        Args:
            style_report: Deterministic StyleReport dict from StyleAnalyzer.
            context_packet: Minimal context projection from ContextCompiler.

        Returns (report_dict, trace).
        """
        mode_hint = ""
        if narrative_mode:
            mode_hint = f"\n当前叙事模式：{narrative_mode}。请根据叙事模式调整评分标准。\n"

        style_section = self._format_style_report(style_report) if style_report else ""
        context_section = self._format_context(context_packet) if context_packet else ""

        content_parts = [
            f"请审查第{chapter_number}章的正文。",
            mode_hint,
        ]
        if context_section:
            content_parts.append(context_section)
        if style_section:
            content_parts.append(style_section)
        content_parts.append(f"## 正文\n{draft_content}")
        content_parts.append("给出完整的审查报告。只输出JSON，不要其他内容。")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]

        content, trace = await self.run_with_tools(
            messages, max_rounds=2, action=f"review_chapter_{chapter_number}"
        )

        defaults = {
            "overall_score": 0,
            "dimensions": {},
            "issues": [],
            "highlights": [],
            "verdict": "manual_review",
        }

        # 与 Continuity 同款：空输出 / 解析失败都标记 unavailable，避免兜底
        # overall_score=0 被进化层误读为「最差版本」触发 crash/regressed：
        #   1. 空输出（reasoning 模型偶发空 content）
        #   2. 非空但解析失败（JSON 语法错误/截断）—— parse_json_response 失败时
        #      会带 raw_output 哨兵字段
        if not (content or "").strip():
            return {"unavailable": True, **defaults}, trace

        raw = parse_json_response(content, defaults=defaults)
        if "raw_output" in raw:
            return {"unavailable": True, **defaults}, trace

        report = OutputValidator.validate("editor", raw).to_dict()
        return report, trace

    @staticmethod
    def _format_style_report(report: dict) -> str:
        """Format StyleReport dict into a prompt section for the Editor."""
        parts = ["## 确定性风格分析（StyleReport）"]

        gate = report.get("style_gate", "PASS")
        parts.append(f"- 结构门禁: {gate}")

        para = report.get("paragraph_structure") or {}
        if para:
            frag = para.get("fragmentation_score", "?")
            short_ratio = para.get("short_narrative_ratio", 0)
            single_ratio = para.get("single_sentence_narrative_ratio", 0)
            max_consec = para.get("max_consecutive_short_narrative_paragraphs", 0)
            para_issues = para.get("issues", [])
            parts.append(
                f"- 段落碎片化评分: {frag}/100, "
                f"短叙述段比例: {short_ratio:.0%}, "
                f"单句段比例: {single_ratio:.0%}, "
                f"最长连续短段: {max_consec}"
            )
            if para_issues:
                parts.append(f"- 段落结构问题: {'; '.join(para_issues)}")

        issues = report.get("issues") or []
        if issues:
            parts.append(f"- AI味证据 ({len(issues)} 项):")
            for i in issues[:8]:
                if isinstance(i, dict):
                    label = i.get("phrase") or i.get("pattern") or i.get("type", "?")
                    parts.append(f"  · {label} (×{i.get('count', 1)})")

        sent = report.get("sentence_rhythm") or {}
        if sent and sent.get("uniform_sentences"):
            parts.append(f"- 句子节奏: {sent.get('detail', '')}")

        ending = report.get("ending_analysis") or {}
        if ending.get("summary_ending"):
            parts.append("- 结尾: 出现总结式结尾（应改为具体动作或画面）")
        hook_ev = ending.get("hook_evidence") or []
        if hook_ev:
            parts.append(f"- 结尾悬念证据: {', '.join(hook_ev)}")

        dialogue = report.get("dialogue_stats") or {}
        if dialogue:
            parts.append(f"- 对话占比: {dialogue.get('detail', '')}")

        return "\n".join(parts)

    @staticmethod
    def _format_context(packet: dict) -> str:
        """Format minimal context projection into a prompt section."""
        parts = []
        if packet.get("recent_summary"):
            parts.append(f"## 前文提要\n{packet['recent_summary']}")
        if packet.get("character_context"):
            parts.append(f"## 相关角色\n{packet['character_context']}")
        if packet.get("unresolved_foreshadowings"):
            fs = packet["unresolved_foreshadowings"]
            parts.append("## 待回收伏笔\n" + "\n".join(f"- {item}" for item in fs))
        return "\n\n".join(parts) if parts else ""
