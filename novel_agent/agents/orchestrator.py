"""Orchestrator Agent — narrative analysis and pipeline context assembly.

Before each chapter, it analyzes the current narrative position and decides
what the chapter needs. Routing decisions are handled by the LangGraph
conditional edges in graph/chapter.py.
"""

from novel_agent.agents.base import AgentConfig, BaseAgent
from novel_agent.schema.validator import parse_validated

ORCHESTRATOR_SYSTEM_PROMPT = """你是一个小说主编，负责统筹整本书的创作方向和节奏。

## 职责

1. **叙事阶段分析**：判断当前阶段并调整策略——intro 建立主角形象与核心冲突；development 推进故事线、扩展世界观；climax 核心冲突爆发；resolution 回收伏笔、收束弧线；unit_arc 独立单元剧情；mini_climax 长高潮阶段的内部波峰；transition 阶段衔接。
2. **篇幅调整**：短篇快速推进、3-5 章内到达 climax；中篇各阶段均衡展开、climax 在总进度 60-70% 处；长篇 intro 充分铺垫、多线并进、伏笔长线回收。
3. **模式感知**：按用户消息的模式指令调整输出——模式决定额外字段及其 schema，均以用户消息为准。

## 输出契约

- 每章必输：narrative_stage、stage_analysis、chapter_strategy（storylines、pacing、foreshadowings_to_address、suggested_chapter_words）、context_needed
- ending_type：仅当本章确需特定结尾类型（如 cliffhanger）时输出，省略即自然收束
- key_scenes / scene_composition / unit_arc / pov_config：仅当用户消息的拆场或模式指令要求时输出
- 其余可选字段仅在确有内容时输出，无内容省略：
  time_structure（非线性叙事）、climax_sequence（climax 阶段）、stage_boundary（阶段边界章）、ending_tone（接近书末/卷末）、storyline_intersection（多线交汇章）、character_arcs（角色里程碑）、character_emotional_state（出场角色显著情绪变化）、tension_profile（紧张度明显起伏）、foreshadowing_management（仅高风险或临近回收的伏笔）

## 通用规则

- storylines 每条含 id、name、role（primary/secondary/tertiary）、progress、chapter_focus（high/medium/low/background）、key_events、status（active/paused/resolved）
- suggested_chapter_words 在目标字数 ±20% 内
- 只输出JSON，不要其他内容
"""

# Conditional schemas — appear exactly once, in the user prompt, and only
# for the mode / scene_first flag the current chapter actually uses.
BASE_OUTPUT_SCHEMA = (
    "输出 schema（本章必输字段）：\n"
    "{\n"
    '  "narrative_stage": "intro|development|climax|resolution|unit_arc|mini_climax|transition",\n'
    '  "stage_analysis": "当前阶段简短分析",\n'
    '  "chapter_strategy": {\n'
    '    "storylines": [{"id": "线ID", "name": "线名", "role": "primary|secondary|tertiary", '
    '"progress": "60%", "chapter_focus": "high|medium|low|background", "key_events": ["重要事件"], '
    '"status": "active|paused|resolved"}],\n'
    '    "pacing": "slow|normal|fast",\n'
    '    "foreshadowings_to_address": ["需要回收或强化的伏笔"],\n'
    '    "suggested_chapter_words": 3000\n'
    "  },\n"
    '  "context_needed": {"characters": ["本章涉及的已有角色"], '
    '"world_elements": ["本章涉及的世界观设定"], "recent_reference": "需要回顾的前文内容", '
    '"cross_timeline_references": [], "perspective_specific": ""}\n'
    "}\n"
)

UNIT_ARC_SCHEMA = (
    '  "unit_arc": {"unit_number": 1, "unit_title": "单元名", '
    '"unit_type": "case_of_the_week|training_arc|filler|character_spotlight", '
    '"mainline_progress": "主线推进", "unit_resolution": "resolved|unresolved|cliffhanger", '
    '"carry_over_elements": []}'
)

POV_CONFIG_SCHEMA = (
    '  "pov_config": {"current_pov": "POV角色", "pov_shift": "视角变化", '
    '"access_level": "surface|moderate|deep", "knowledge_gap": "视角信息差"}'
)

KEY_SCENES_SCHEMA = '  "key_scenes": ["场景名·地点·冲突核心·情绪落点", ...]  // 3-4 个分镜'

SCENE_COMPOSITION_SCHEMA = (
    '  "scene_composition": {"primary_scene_type": '
    '"action|dialogue|introspection|description|transition|mixed", '
    '"scene_breakdown": "场景构成", "recent_dominance": "近章主导类型", '
    '"diversity_warning": "同质化警告"}'
)


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)

    @property
    def system_prompt(self) -> str:
        return ORCHESTRATOR_SYSTEM_PROMPT

    async def analyze(
        self,
        chapter_number: int,
        chapter_outline: str,
        previous_chapters: list[dict],
        story_length: str = "long",
        target_chapter_words: int = 3000,
        narrative_mode: str | None = None,
        narrative_perspective: str = "",
        arc_summary: str = "",
        context_packet: dict | None = None,
        total_chapters: int = 0,
        scene_first: bool = False,
    ) -> dict:
        """Analyze narrative position and decide chapter strategy.

        Args:
            previous_chapters: Recent chapters (ascending) — a tail slice, not
                the full history.
            total_chapters: Total completed chapters; falls back to
                len(previous_chapters) when not provided.
            scene_first: Whether this chapter is generated scene-by-scene;
                controls whether key_scenes / scene_composition are required.

        Returns a dict with narrative_stage, chapter_strategy, context_needed.
        """
        total = total_chapters or len(previous_chapters)

        recent = previous_chapters[-3:] if len(previous_chapters) > 3 else previous_chapters
        recent_titles = ", ".join(f"第{c.get('chapter_number', '?')}章" for c in recent)

        length_label = {
            "short": "短篇",
            "medium": "中篇",
            "long": "长篇",
        }.get(story_length, story_length or "长篇")

        packet = context_packet or {}
        character_context = packet.get("character_context", "")
        world_context = packet.get("world_context", "")
        recent_summary = packet.get("recent_summary", "")
        unresolved_foreshadowings = packet.get("unresolved_foreshadowings", [])
        timeline_events = packet.get("timeline_events", [])
        timeline_findings = packet.get("timeline_findings", [])
        unresolved = unresolved_foreshadowings or []
        foreshadowing_context = "\n".join(f"- {item}" for item in unresolved)
        timeline_context = "\n".join(str(item) for item in (timeline_events or [])[-10:])
        timeline_warnings = "\n".join(str(item) for item in (timeline_findings or [])[:10])

        # Skip-when-empty: absent sections stay absent, no placeholder noise.
        summary_section = f"## 前情摘要\n{recent_summary}\n\n" if recent_summary else ""

        mode_instruction = self._build_mode_instruction(narrative_mode)
        persp_hint = self._build_perspective_hint(narrative_perspective)
        scene_instruction = self._build_scene_instruction(scene_first)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请分析当前叙事状态并制定第{chapter_number}章的策略。\n\n"
                    f"{mode_instruction}\n"
                    f"{persp_hint}"
                    f"{scene_instruction}"
                    f"{BASE_OUTPUT_SCHEMA}\n"
                    f"## 篇幅信息\n"
                    f"- 篇幅：{length_label}\n"
                    f"- 目标每章字数：{target_chapter_words}字\n"
                    f"- 已完成章节数：{total}章\n\n"
                    f"## 本章大纲\n{chapter_outline}\n\n"
                    f"## 已有角色\n{character_context or '暂无'}\n\n"
                    f"## 世界观设定\n{world_context or '暂无'}\n\n"
                    f"## 已有章节\n{recent_titles or '无'}\n\n"
                    f"{summary_section}"
                    f"{arc_summary}\n\n"
                    f"## 待回收伏笔\n{foreshadowing_context or '暂无'}\n\n"
                    f"## 关键事件\n{timeline_context or '暂无'}\n\n"
                    f"## 时间线警告\n{timeline_warnings or '暂无'}\n\n"
                ),
            },
        ]

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"orchestrate_ch{chapter_number}",
        )

        result = parse_validated(
            "orchestrator",
            content,
            defaults={
                "narrative_stage": "development",
                "stage_analysis": "",
                "chapter_strategy": {
                    "storylines": [],
                    "pacing": "normal",
                    "key_scenes": [],
                    "foreshadowings_to_address": [],
                    "suggested_chapter_words": target_chapter_words,
                    "climax_sequence": None,
                    "stage_boundary": None,
                    "unit_arc": None,
                    "pov_config": None,
                    "time_structure": {"mode": "linear", "current_timeline": "present"},
                    "ending_tone": None,
                    "storyline_intersection": None,
                    "character_arcs": [],
                    "scene_composition": None,
                },
                "context_needed": {
                    "characters": [],
                    "world_elements": [],
                    "recent_reference": "",
                    "cross_timeline_references": [],
                    "perspective_specific": "",
                },
            },
        )

        return result

    @staticmethod
    def _build_mode_instruction(mode: str | None) -> str:
        """Mode-aware conditional schema — lives in the user prompt only.

        The system prompt carries generic output rules; each mode's field
        structure appears exactly once, here, and only for the active mode.
        Empty string when the mode is not configured.
        """
        if not mode:
            return ""

        base = f"当前叙事模式：{mode}\n"
        if mode in ("unit_arc", "hybrid"):
            return (
                f"{base}本章 chapter_strategy 必须额外输出（结构如下）：\n"
                f"{UNIT_ARC_SCHEMA}\n"
                "其余可选字段按需输出，无内容省略。\n"
            )
        if mode in ("multi_perspective", "ensemble"):
            return (
                f"{base}本章 chapter_strategy 必须额外输出（结构如下）：\n"
                f"{POV_CONFIG_SCHEMA}\n"
                "其余可选字段按需输出，无内容省略。\n"
            )
        # linear and other modes: only the always-required fields.
        return base + "本章只需输出必输字段与确有内容的可选字段，不要输出空的可选字段。\n"

    @staticmethod
    def _build_scene_instruction(scene_first: bool) -> str:
        """Scene-first conditional schema — user prompt only."""
        if scene_first:
            return (
                "本章为 scene_first 拆场模式：必须将本章拆解为 3-4 个分镜场景，"
                "chapter_strategy 额外输出（结构如下）：\n"
                f"{KEY_SCENES_SCHEMA}\n"
                f"{SCENE_COMPOSITION_SCHEMA}\n"
            )
        return (
            "本章为整章生成模式：不需要输出 key_scenes 与 scene_composition，"
            "将场景安排融入 storylines 的 key_events 中。\n"
        )

    @staticmethod
    def _build_perspective_hint(perspective: str) -> str:
        """Build perspective constraint hint for the Orchestrator prompt.

        Tells the Orchestrator about the narrator's information access limits,
        so context_needed and chapter_strategy respect POV constraints.
        """
        if not perspective:
            return ""

        hints = {
            "first_person": (
                "叙事视角约束：第一人称。context_needed 中的角色和世界观信息"
                "应限定为主角所能感知的范围。主角不知道的事情不应出现在"
                "perspective_specific 之外。\n"
            ),
            "third_person_limited": (
                "叙事视角约束：第三人称受限。聚焦单一角色的认知范围，"
                "可在 perspective_specific 中标注该角色不知道的关键信息。\n"
            ),
            "third_person_omniscient": (
                "叙事视角约束：第三人称全知。无信息访问限制，"
                "context_needed 可包含任意角色的信息。\n"
            ),
        }

        return hints.get(perspective, "")
