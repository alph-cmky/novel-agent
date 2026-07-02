"""Orchestrator Agent — narrative analysis and pipeline context assembly.

Before each chapter, it analyzes the current narrative position and decides
what the chapter needs. Routing decisions are handled by the LangGraph
conditional edges in graph/chapter.py.
"""

from novel_agent.agents.base import AgentConfig, BaseAgent
from novel_agent.memory.compressor import ContextCompressor
from novel_agent.schema.parser import parse_json_response

ORCHESTRATOR_SYSTEM_PROMPT = """你是一个小说主编，负责统筹整本书的创作方向和节奏。

## 你的职责

### 1. 叙事阶段分析
判断当前故事处于哪个阶段，据此调整创作策略：
- **intro（开篇）**：建立主角形象，展示金手指，埋下核心冲突
- **development（发展）**：推进故事线，扩展世界观，发展角色关系
- **climax（高潮）**：核心冲突爆发，关键抉择，情绪高点
- **resolution（收尾）**：回收伏笔，角色弧线完成，留有余味

### 2. 篇幅调整
根据用户选择的篇幅调整节奏：
- **短篇**：快速推进，跳过intro直接进入development，3-5章内到达climax
- **中篇**：平衡发展，各阶段充分展开，climax在总进度60-70%处
- **长篇**：渐进展开，intro充分铺垫，development多线并进，伏笔长线回收

### 3. 章节策略决策
根据阶段分析和篇幅，决定本章需要：
- 推进哪条故事线（主线/感情线/悬疑线）
- 需要什么节奏（快节奏战斗/慢节奏情感/信息揭露）
- 应该在高点还是悬念处结束
- 需要回收哪些伏笔

## 输出格式

```json
{
  "narrative_stage": "intro|development|climax|resolution",
  "stage_analysis": "当前阶段的简短分析",
  "chapter_strategy": {
    "primary_storyline": "主要推进的故事线",
    "pacing": "建议节奏",
    "key_scenes": ["本章必须包含的场景"],
    "ending_type": "cliffhanger|emotional_beat|revelation",
    "foreshadowings_to_address": ["需要回收或强化的伏笔"],
    "suggested_chapter_words": 3000
  },
  "context_needed": {
    "characters": ["本章涉及的已有角色"],
    "world_elements": ["本章涉及的世界观设定"],
    "recent_reference": "需要回顾的前文内容描述"
  }
}
```

只输出JSON。
"""


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"

    def __init__(
        self,
        config: AgentConfig | None = None,
        compressor: ContextCompressor | None = None,
    ):
        super().__init__(config)
        self._compressor = compressor or ContextCompressor()
        self._story_arc: list[dict] = []  # Chapter-by-chapter narrative tracking

    @property
    def system_prompt(self) -> str:
        return ORCHESTRATOR_SYSTEM_PROMPT

    async def analyze(
        self,
        chapter_number: int,
        chapter_outline: str,
        previous_chapters: list[dict],
        character_context: str,
        world_context: str,
        story_length: str = "long",
        target_chapter_words: int = 3000,
    ) -> dict:
        """Analyze narrative position and decide chapter strategy.

        Returns a dict with narrative_stage, chapter_strategy, context_needed.
        """
        total_chapters = len(previous_chapters)

        # Build arc summary from tracked data
        arc_summary = ""
        if self._story_arc:
            arc_entries = [
                f"第{a.get('chapter', '?')}章: {a.get('stage', '?')} - "
                f"Editor {a.get('editor_score', '?')}/100, "
                f"Continuity {a.get('continuity_score', '?')}/100"
                for a in self._story_arc[-5:]
            ]
            arc_summary = "## 最近章节表现\n" + "\n".join(arc_entries)

        recent = previous_chapters[-3:] if len(previous_chapters) > 3 else previous_chapters
        recent_titles = ", ".join(
            f"第{c.get('chapter_number', '?')}章" for c in recent
        )

        length_label = {"short": "短篇", "novella": "中篇", "long": "长篇"}.get(
            story_length, "长篇"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请分析当前叙事状态并制定第{chapter_number}章的策略。\n\n"
                    f"## 篇幅信息\n"
                    f"- 篇幅：{length_label}\n"
                    f"- 目标每章字数：{target_chapter_words}字\n"
                    f"- 已完成章节数：{total_chapters}章\n\n"
                    f"## 本章大纲\n{chapter_outline}\n\n"
                    f"## 已有角色\n{character_context or '暂无'}\n\n"
                    f"## 世界观设定\n{world_context or '暂无'}\n\n"
                    f"## 已有章节\n{recent_titles or '无'}\n\n"
                    f"{arc_summary}\n\n"
                    f"只输出JSON。"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"orchestrate_ch{chapter_number}",
        )

        result = parse_json_response(content, defaults={
            "narrative_stage": "development",
            "stage_analysis": "",
            "chapter_strategy": {
                "primary_storyline": "",
                "pacing": "normal",
                "key_scenes": [],
                "ending_type": "cliffhanger",
                "foreshadowings_to_address": [],
                "suggested_chapter_words": target_chapter_words,
            },
            "context_needed": {
                "characters": [], "world_elements": [], "recent_reference": "",
            },
        })

        # Track in story arc
        self._story_arc.append({
            "chapter": chapter_number,
            "stage": result.get("narrative_stage", "?"),
            "strategy": result.get("chapter_strategy", {}),
        })

        return result

    async def review_feedback(
        self,
        chapter_number: int,
        chapter_outline: str,
        draft_content: str,
        editor_report: dict,
        continuity_report: dict,
        human_feedback: dict | None = None,
    ) -> str:
        """Analyze Editor/Continuity/Human feedback and produce rewrite instructions.

        Called when the pipeline detects issues — either automatically (scores low)
        or when a human reviewer rejects the draft. Produces specific, actionable
        guidance for the Writer to fix the identified problems.
        """
        editor_score = editor_report.get("overall_score", 0)
        continuity_score = continuity_report.get("overall_score", 0)

        # Summarize key issues for the prompt
        editor_issues = editor_report.get("issues", [])[:5]
        editor_issues_text = "\n".join(
            f"- [{i.get('severity', '?')}] [{i.get('category', '?')}] "
            f"{i.get('description', '')}\n  Suggestion: {i.get('suggestion', 'N/A')}"
            for i in editor_issues
        ) or "无"

        continuity_issues = continuity_report.get("inconsistencies", [])[:5]
        continuity_issues_text = "\n".join(
            f"- [{i.get('severity', '?')}] [{i.get('category', '?')}] "
            f"{i.get('description', '')}\n  Current: {i.get('current', '?')}\n  "
            f"Previous: {i.get('previous', '?')}"
            for i in continuity_issues
        ) or "无"

        human_note = ""
        if human_feedback:
            action = human_feedback.get("action", "?")
            comments = human_feedback.get("comments", "")
            edited = human_feedback.get("edited_text", "")
            human_note = f"## 人类审阅反馈\n- 决定: {action}\n- 意见: {comments}\n"
            if edited:
                human_note += f"- 编辑文本参考: {edited[:500]}\n"

        messages = [
            {"role": "system", "content": (
                "你是小说主编。你的任务是分析审阅反馈并给出具体的重写指导。\n\n"
                "## 分析要点\n"
                "1. 哪些是结构性问题（逻辑矛盾、角色不一致）→ Writer 必须修正\n"
                "2. 哪些是风格问题（AI味、节奏、对话）→ Writer 应该改进\n"
                "3. 哪些是低优先级问题 → 可暂不处理\n\n"
                "## 输出格式\n"
                "直接输出重写指导（纯文本），格式：\n"
                "```\n"
                "## 必须修正（结构性问题）\n"
                "1. [具体问题] → [如何修正]\n"
                "2. ...\n\n"
                "## 应该改进（风格/质量问题）\n"
                "1. [具体问题] → [如何改进]\n"
                "2. ...\n\n"
                "## 保持\n"
                "[本次写得好的部分，重写时保留]\n"
                "```"
            )},
            {"role": "user", "content": (
                f"第{chapter_number}章需要重写。请分析以下反馈并给出重写指导。\n\n"
                f"## 本章大纲\n{chapter_outline}\n\n"
                f"## 评分\n- Editor: {editor_score}/100\n"
                f"- Continuity: {continuity_score}/100\n\n"
                f"## Editor 问题\n{editor_issues_text}\n\n"
                f"## Continuity 问题\n{continuity_issues_text}\n\n"
                f"{human_note}"
                f"## 当前草稿（前800字）\n{draft_content[:800]}\n"
            )},
        ]

        content, _ = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"review_feedback_ch{chapter_number}",
        )
        return content.strip()

    def get_arc_summary(self) -> list[dict]:
        """Return the story arc tracking data."""
        return self._story_arc
