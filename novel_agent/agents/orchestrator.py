"""Orchestrator Agent — narrative analysis and pipeline context assembly.

Before each chapter, it analyzes the current narrative position and decides
what the chapter needs. Routing decisions are handled by the LangGraph
conditional edges in graph/chapter.py.
"""

from novel_agent.agents.base import AgentConfig, BaseAgent
from novel_agent.schema.validator import parse_validated

ORCHESTRATOR_SYSTEM_PROMPT = """你是一个小说主编，负责统筹整本书的创作方向和节奏。

## 你的职责

### 1. 叙事阶段分析
判断当前故事处于哪个阶段，据此调整创作策略：
- **intro（开篇）**：建立主角形象，展示金手指，埋下核心冲突
- **development（发展）**：推进故事线，扩展世界观，发展角色关系
- **climax（高潮）**：核心冲突爆发，关键抉择，情绪高点
- **resolution（收尾）**：回收伏笔，角色弧线完成，留有余味
- **unit_arc（单元剧）**：独立单元剧情，主线松散串联
- **mini_climax（小高潮）**：长高潮阶段的内部起伏波峰
- **transition（过渡）**：阶段之间的衔接章节

### 2. 篇幅调整
根据用户选择的篇幅调整节奏：
- **短篇**：快速推进，跳过intro直接进入development，3-5章内到达climax
- **中篇**：平衡发展，各阶段充分展开，climax在总进度60-70%处
- **长篇**：渐进展开，intro充分铺垫，development多线并进，伏笔长线回收

### 3. 叙事模式感知
根据项目的 narrative_mode 调整输出：
- **linear（线性）**：按单一主线推进，storylines 中 role=primary 的一条
- **unit_arc（单元剧）**：输出 unit_arc 字段，主线在单元之间松散串联
- **hybrid（混合）**：单元剧情 + 主线推进，同时输出 storylines 和 unit_arc
- **multi_perspective（多视角）**：输出 pov_config 字段，指定每章的 POV 角色
- **ensemble（群像）**：多线并行，pov_config 标记聚焦角色，storylines 管理各线进度

## 输出字段分层（按需输出，不要每章输出全部字段）

**每章必须输出**：storylines、pacing、foreshadowings_to_address、suggested_chapter_words

**ending_type**：仅当本章确需特定结尾类型（如 cliffhanger、revelation）时输出；不输出则自然收束

**仅 scene_first 拆场模式输出**：key_scenes（3-4 个分镜：场景名·地点·冲突核心·情绪落点）、scene_composition

**仅对应叙事模式输出**（见用户消息中的模式指令）：unit_arc（unit_arc/hybrid 模式）、pov_config（multi_perspective/ensemble 模式）

**可选字段**（仅在确有内容时输出，无内容省略或置 null）：
- time_structure {mode(linear|flashback|parallel|non_linear), current_timeline, flashback_trigger, time_gap, synchronization_points}：非线性叙事时
- climax_sequence {current_mini_climax, total_mini_climaxes, mini_climax_type, previous_resolution}：climax 阶段 + 长篇
- stage_boundary {is_boundary, boundary_type(world_switch|time_jump|conflict_escalation|resolution_start), previous_stage_duration, estimated_next_stage_duration}：阶段边界章
- ending_tone {type, ambiguity_level(high|medium|low), reader_satisfaction, setup_for_next}：接近全书或大卷结尾
- storyline_intersection {has_intersection, intersection_type(crossover|parallel|contrast), intersection_description}：多线交汇章
- character_arcs [{character_name, arc_stage(growth|fall|redemption|flat|corruption), arc_progress, current_state, recent_milestone, next_milestone}]：角色里程碑事件
- character_emotional_state {角色名: {mood, trigger, intensity}}：出场角色显著情绪变化
- tension_profile {chapter_tension(1-10), overall_trend(rising|falling|holding|peak|valley), recent_chapters_tension[], emotional_tone, variety_check}：紧张度明显起伏
- foreshadowing_management [{description, chapters_outstanding, risk_level(low|medium|high), action_needed(tease|advance|resolve|maintain), reader_knows, characters_aware[], characters_unaware[]}]：仅高风险或临近回收的伏笔
- unit_arc {unit_number, unit_title, unit_type(case_of_the_week|training_arc|filler|character_spotlight), mainline_progress, unit_resolution(resolved|unresolved|cliffhanger), carry_over_elements[]}：unit_arc/hybrid 模式必输
- pov_config {current_pov, pov_shift, access_level(surface|moderate|deep), knowledge_gap}：multi_perspective/ensemble 模式必输
- scene_composition {primary_scene_type(action|dialogue|introspection|description|transition|mixed), scene_breakdown, recent_dominance, diversity_warning}：scene_first 模式必输

## 输出格式

```json
{
  "narrative_stage": "intro|development|climax|resolution|unit_arc|mini_climax|transition",
  "stage_analysis": "当前阶段的简短分析",
  "chapter_strategy": {
    "storylines": [
      {"id": "线ID", "name": "线名", "role": "primary|secondary|tertiary", "progress": "60%",
       "chapter_focus": "high|medium|low|background", "key_events": ["重要事件"], "status": "active|paused|resolved"}
    ],
    "pacing": "slow|normal|fast",
    "foreshadowings_to_address": ["需要回收或强化的伏笔"],
    "suggested_chapter_words": 3000,
    "key_scenes": ["scene_first 模式必填：分镜场景"]
  },
  "context_needed": {
    "characters": ["本章涉及的已有角色"],
    "world_elements": ["本章涉及的世界观设定"],
    "recent_reference": "需要回顾的前文内容描述",
    "cross_timeline_references": [],
    "perspective_specific": ""
  }
}
```

chapter_strategy 其余字段（unit_arc / pov_config / time_structure / climax_sequence / stage_boundary / ending_tone / storyline_intersection / character_arcs / character_emotional_state / tension_profile / foreshadowing_management / scene_composition）按上方分层要求按需输出。

只输出JSON。
"""


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
                    f"只输出JSON。"
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
                    "character_emotional_state": {},
                    "tension_profile": None,
                    "foreshadowing_management": [],
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
        """Build mode-specific output instructions for the Orchestrator prompt.

        Field structures live once in the system prompt; the mode instruction
        only names the fields this mode must emit. Empty string when the mode
        is not configured.
        """
        if not mode:
            return ""

        base = f"当前叙事模式：{mode}\n"
        required = {
            "unit_arc": "unit_arc",
            "hybrid": "unit_arc",
            "multi_perspective": "pov_config",
            "ensemble": "pov_config",
        }.get(mode)
        if required:
            return f"{base}本章必须输出 {required} 字段（结构见系统提示），其余可选字段按需输出，无内容省略。\n"
        # linear and other modes: only the always-required fields.
        return base + "本章只需输出必输字段与确有内容的可选字段，不要输出空的可选字段。\n"

    @staticmethod
    def _build_scene_instruction(scene_first: bool) -> str:
        """Dynamic key_scenes requirement — only meaningful for scene_first runs."""
        if scene_first:
            return (
                "本章为 scene_first 拆场模式：必须将本章拆解为 3-4 个分镜场景"
                "输出 key_scenes，并输出 scene_composition。\n"
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
