"""Writer Agent — generates chapter content from outline and context."""

from collections.abc import AsyncIterator

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.parser import strip_none
from novel_agent.tools.search import SearchContextTool

WRITER_SYSTEM_PROMPT = """你是一个顶级长篇网文作家，擅长创作极具张力、画面感强、情绪流与节奏明快的精彩正文。

## 核心创作法则（网文黄金标准）

1. **动作与画面推进（Show, Don't Tell）**：
   - 拒绝抽象说明。不要写"他很震惊/愤怒"，写具体的生理反应与微动作（如"他手指猛地收紧，骨节泛白"、"茶杯在掌心生生捏出裂纹"）。
   - 拒绝虚浮的形容词堆砌，多用强动词、生活化口语与短句，提升阅读爽感。

2. **高频对抗与对话张力（对白占比 40%+）**：
   - 每一段对白都必须带有目的、潜台词或情绪交锋，严禁无营养的客套和问答。
   - 角色说话必须符合其身份、地位与性格，杜绝所有角色千人一面的"书面播音腔"。

3. **感官沉浸（五感细节）**：
   - 每场戏至少融入 1~2 处精准的视觉（光影/色彩）、听觉（环境声/破空声）或触觉/嗅觉描写，营造身临其境的现场感。

4. **结尾断章与悬念钩子（Cliffhanger）**：
   - 严格禁止在章节末尾做任何道德总结、中心思想升华或大段哲学感慨！
   - 必须戛然而止在关键动作、突发异变、人物意外登场或致命悬念对白上，让读者迫不及待点击下一章。

5. **硬性去 AI 味法则**：
   - **绝对禁用词**：此外、不仅如此、更重要的是、至关重要、不可忽视、显而易见、总而言之、由此可见、基于以上。
   - **绝对禁用套话**：禁止写"他的眼中闪过一丝..."、"嘴角勾起一抹弧度"、"深吸一口气，平复心情"、"这不仅是...更是..."。
   - **句式多样化**：长句与短句交错，拒绝连续三个长度相同的匀速排比句。

6. **篇幅与结构完整性**：
   - 必须输出包含起承转合的完整正文章节（禁止概括性大纲、禁止只输出修改片段或省略号）。
   - 充分展开场景与对话，严禁过度压缩剧情。

## 策略指令解读
主编（Orchestrator）在 prompt 中提供的叙事策略与关键场景必须严格覆盖并自然融入剧情。

## 输出格式
直接输出小说章节正文，不要输出任何前言、后记、写作说明或多余标题。每章2000-4000字。"""


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(
        self,
        config: AgentConfig | None = None,
        chapter_store: ChapterStore | None = None,
        project_id: str = "",
        target_chapter_words: int = 3000,
        narrative_mode: str | None = None,
        narrative_perspective: str = "",
    ):
        super().__init__(config)
        self._chapter_store = chapter_store
        self._project_id = project_id
        self._target_words = target_chapter_words
        self._narrative_mode = narrative_mode
        self._narrative_perspective = narrative_perspective
        if chapter_store and project_id:
            self.register_tool(SearchContextTool(chapter_store, project_id))

    @property
    def system_prompt(self) -> str:
        prompt = WRITER_SYSTEM_PROMPT
        if self._target_words:
            prompt = prompt.replace(
                "每章2000-4000字",
                f"每章{self._target_words}字左右",
            )
        return prompt

    async def write(
        self,
        chapter_number: int,
        outline: str,
        character_context: str = "",
        world_context: str = "",
        recent_summary: str = "",
        target_chapter_words: int = 0,
        rewrite_instructions: str = "",
        orchestrator_strategy: dict | None = None,
        unresolved_foreshadowings: list[str] | None = None,
    ) -> tuple[str, TraceStep]:
        """Generate a chapter.

        Args:
            chapter_number: Current chapter number.
            outline: Chapter outline / plot points.
            character_context: Relevant character info from memory.
            world_context: Relevant worldbuilding info from memory.
            recent_summary: Compressed summary of recent chapters.
            target_chapter_words: Override per-chapter word count (0 = use default).
            rewrite_instructions: Specific guidance from Orchestrator for rewriting.
            orchestrator_strategy: Narrative strategy from Orchestrator (stage, pacing, etc.).
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Assemble context
        context_parts = [f"## 第{chapter_number}章大纲\n{outline}"]
        if rewrite_instructions:
            context_parts.insert(0, f"## 重写指导（务必遵守）\n{rewrite_instructions}")
        if orchestrator_strategy:
            strategy_text = self._format_strategy(orchestrator_strategy)
            if strategy_text:
                context_parts.insert(1 if rewrite_instructions else 0, strategy_text)
        if character_context:
            context_parts.append(f"## 相关角色\n{character_context}")
        if world_context:
            context_parts.append(f"## 世界观设定\n{world_context}")
        if recent_summary:
            context_parts.append(f"## 前文提要\n{recent_summary}")
        if unresolved_foreshadowings:
            context_parts.append(
                "## 待回收伏笔（不得无故遗忘或提前泄露）\n"
                + "\n".join(f"- {item}" for item in unresolved_foreshadowings)
            )

        # 仅当 search_context 工具已注册（project_id 非空）才提示使用工具；
        # 否则（评测/无项目库场景）模型无法调用工具，会输出 <search_context> 文本
        # 标签并提前终止，产出空正文。
        tool_hint = (
            "创作前请先使用 search_context 工具检索关键信息。"
            if self._tools
            else "直接输出章节正文，不要加任何说明。"
        )
        user_prompt = (
            f"请根据以下信息创作第{chapter_number}章：\n\n"
            + "\n\n".join(context_parts)
            + f"\n\n{tool_hint}"
        )
        messages.append({"role": "user", "content": user_prompt})

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=3,
            action=f"write_chapter_{chapter_number}",
        )
        return content, trace

    async def write_stream(
        self,
        chapter_number: int,
        outline: str,
        character_context: str = "",
        world_context: str = "",
        recent_summary: str = "",
        target_chapter_words: int = 0,
        rewrite_instructions: str = "",
        orchestrator_strategy: dict | None = None,
        unresolved_foreshadowings: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a chapter with streaming output. Yields text chunks.

        Unlike write(), this skips tool calling and streams the LLM response directly.
        """
        context_parts = [f"## 第{chapter_number}章大纲\n{outline}"]
        if rewrite_instructions:
            context_parts.insert(0, f"## 重写指导（务必遵守）\n{rewrite_instructions}")
        if orchestrator_strategy:
            strategy_text = self._format_strategy(orchestrator_strategy)
            if strategy_text:
                context_parts.insert(1 if rewrite_instructions else 0, strategy_text)
        if character_context:
            context_parts.append(f"## 相关角色\n{character_context}")
        if world_context:
            context_parts.append(f"## 世界观设定\n{world_context}")
        if recent_summary:
            context_parts.append(f"## 前文提要\n{recent_summary}")
        if unresolved_foreshadowings:
            context_parts.append(
                "## 待回收伏笔（不得无故遗忘或提前泄露）\n"
                + "\n".join(f"- {item}" for item in unresolved_foreshadowings)
            )

        user_prompt = (
            f"请根据以下信息创作第{chapter_number}章：\n\n"
            + "\n\n".join(context_parts)
            + "\n\n直接输出章节正文，不要加任何说明。"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async for chunk in self.call_model_stream(
            messages,
            action=f"write_stream_ch{chapter_number}",
        ):
            yield chunk

    def _format_strategy(self, strategy: dict) -> str:
        """Format orchestrator strategy dict into a three-tier prompt section.

        Tier 1 (GLOBAL): Stage info + core strategy — always injected.
        Tier 2 (CONDITIONAL): Mode-specific guidance — only when narrative_mode matches.
        Tier 3 (AUXILIARY): Experience hints — injected as [参考] suggestions.
        """
        cs = strategy.get("chapter_strategy") or {}
        if not isinstance(cs, dict):
            cs = {}
        cs = strip_none(cs)
        parts = []

        # ── Stage context (from top-level strategy, not chapter_strategy) ──
        stage = strategy.get("narrative_stage", "")
        analysis = strategy.get("stage_analysis", "")
        if stage or analysis:
            stage_label = {
                "intro": "开篇", "development": "发展",
                "climax": "高潮", "resolution": "收尾",
                "unit_arc": "单元剧", "mini_climax": "小高潮",
                "transition": "过渡",
            }.get(stage, stage)
            lines = ["## 叙事策略（主编指导）"]
            lines.append(f"- 当前阶段：{stage_label}")
            if analysis:
                lines.append(f"- 阶段分析：{analysis}")
            parts.append("\n".join(lines))

        # ── Tier 1: GLOBAL — always injected ──
        global_section = self._format_global_section(cs)
        if global_section:
            parts.append(global_section)

        # ── Tier 2: CONDITIONAL — mode-specific ──
        if self._narrative_mode:
            cond_section = self._format_conditional_section(cs)
            if cond_section:
                parts.append(cond_section)

        # ── Tier 3: AUXILIARY — experience hints ──
        aux_section = self._format_auxiliary_section(cs)
        if aux_section:
            parts.append(aux_section)

        return "\n\n".join(parts)

    # ── Tier formatters ────────────────────────────────────

    @staticmethod
    def _format_global_section(cs: dict) -> str:
        """GLOBAL fields: core strategy every chapter needs."""
        lines = ["## 本章战略"]
        has_content = False

        # Storylines focus
        storylines = cs.get("storylines", [])
        focused = [s for s in storylines if s.get("chapter_focus") in ("high", "medium")]
        if focused:
            lines.append("### 故事线")
            for s in focused:
                lines.append(
                    f"- {s.get('name', '?')}（聚焦:{s.get('chapter_focus', '?')}, "
                    f"进度:{s.get('progress', '?')}）"
                )
            has_content = True

        # Key scenes
        if cs.get("key_scenes"):
            lines.append("### 关键场景")
            for scene in cs["key_scenes"]:
                lines.append(f"- {scene}")
            has_content = True

        # Pacing
        if cs.get("pacing"):
            lines.append(f"### 节奏: {cs['pacing']}")
            has_content = True

        # Ending type
        if cs.get("ending_type"):
            lines.append(f"### 结尾: {cs['ending_type']}")
            has_content = True

        # Foreshadowings to address (fixes 改进点9)
        foreshadowings = cs.get("foreshadowings_to_address", [])
        if foreshadowings:
            lines.append("### 需处理的伏笔")
            for f in foreshadowings:
                lines.append(f"- {f}")
            has_content = True

        # Suggested chapter words (fixes 改进点10)
        if cs.get("suggested_chapter_words"):
            lines.append(
                f"### 建议字数: {cs['suggested_chapter_words']}字"
            )
            has_content = True

        return "\n".join(lines) if has_content else ""

    def _format_conditional_section(self, cs: dict) -> str:
        """CONDITIONAL fields: only injected when narrative_mode matches."""
        lines = ["## 模式特定指导"]
        has_content = False
        mode = self._narrative_mode or ""

        # climax_sequence: only in climax stage
        if cs.get("climax_sequence"):
            cq = cs["climax_sequence"]
            lines.append(
                f"- 高潮序列：第{cq.get('current_mini_climax', '?')}/"
                f"{cq.get('total_mini_climaxes', '?')}个小高潮"
                f"（{cq.get('mini_climax_type', '?')}）"
            )
            if cq.get("previous_resolution"):
                lines.append(f"  承接：{cq['previous_resolution']}")
            has_content = True

        # unit_arc: only unit_arc / hybrid mode
        if mode in ("unit_arc", "hybrid") and cs.get("unit_arc"):
            ua = cs["unit_arc"]
            lines.append(
                f"- 单元模式：第{ua.get('unit_number', '?')}单元"
                f"「{ua.get('unit_title', '?')}」({ua.get('unit_type', '?')})"
            )
            lines.append(f"  主线推进：{ua.get('mainline_progress', '?')}")
            if ua.get("carry_over_elements"):
                lines.append(
                    f"  跨单元线索：{'；'.join(ua.get('carry_over_elements', []))}"
                )
            has_content = True

        # pov_config: only multi_perspective / ensemble mode
        if mode in ("multi_perspective", "ensemble") and cs.get("pov_config"):
            persp = cs["pov_config"]
            if persp.get("current_pov"):
                lines.append(
                    f"- 当前POV：{persp['current_pov']}"
                    f"（{persp.get('access_level', 'deep')}）"
                )
            if persp.get("knowledge_gap"):
                lines.append(f"- 视角信息差：{persp['knowledge_gap']}")
            has_content = True

        # time_structure: only when non-linear
        ts = cs.get("time_structure") or {}
        if ts.get("mode") and ts["mode"] != "linear":
            lines.append(
                f"- 时间结构：{ts['mode']}，"
                f"当前时间线：{ts.get('current_timeline', 'present')}"
            )
            if ts.get("flashback_trigger"):
                lines.append(f"  倒叙触发：{ts['flashback_trigger']}")
            has_content = True

        # stage_boundary: only when boundary detected
        if cs.get("stage_boundary", {}).get("is_boundary"):
            sb = cs["stage_boundary"]
            lines.append(f"- 阶段边界：{sb.get('boundary_type', '?')}")
            lines.append(
                f"  前阶段持续了{sb.get('previous_stage_duration', '?')}章"
            )
            has_content = True

        # ending_tone: near resolution / ending
        if cs.get("ending_tone"):
            et = cs["ending_tone"]
            lines.append(
                f"- 结尾风格：{et.get('type', '?')}"
                f"（歧义度:{et.get('ambiguity_level', 'medium')}）"
            )
            has_content = True

        # storyline_intersection: when multiple lines cross
        if cs.get("storyline_intersection", {}).get("has_intersection"):
            si = cs["storyline_intersection"]
            lines.append(
                f"- 故事线交汇：{si.get('intersection_type', '?')}"
                f" — {si.get('intersection_description', '')}"
            )
            has_content = True

        return "\n".join(lines) if has_content else ""

    @staticmethod
    def _format_auxiliary_section(cs: dict) -> str:
        """AUXILIARY fields: always injected as [参考] hints."""
        lines = ["## 创作参考（以下为指导建议，以文学效果为先）"]
        has_content = False

        # Tension profile
        if cs.get("tension_profile"):
            tp = cs["tension_profile"]
            lines.append(
                f"- 本章紧张度：{tp.get('chapter_tension', '?')}/10，"
                f"趋势：{tp.get('overall_trend', '?')}"
            )
            lines.append(f"- 情绪基调：{tp.get('emotional_tone', '?')}")
            vc = tp.get("variety_check", {})
            if vc.get("suggestion"):
                lines.append(f"- 节奏提醒：{vc['suggestion']}")
            has_content = True

        # Scene composition
        if cs.get("scene_composition"):
            sc = cs["scene_composition"]
            lines.append(
                f"- 建议主类型：{sc.get('primary_scene_type', '?')}"
            )
            if sc.get("diversity_warning"):
                lines.append(f"- 多样性提醒：{sc['diversity_warning']}")
            has_content = True

        # Character emotional state
        if cs.get("character_emotional_state"):
            lines.append("- 角色情绪：")
            for name, state in cs["character_emotional_state"].items():
                lines.append(
                    f"  {name}: {state.get('mood', '?')}"
                    f"（{state.get('trigger', '')}，"
                    f"强度:{state.get('intensity', '?')}）"
                )
            has_content = True

        # Character arcs
        if cs.get("character_arcs"):
            lines.append("- 角色弧线：")
            for arc in cs["character_arcs"]:
                lines.append(
                    f"  {arc.get('character_name', '?')}: "
                    f"{arc.get('current_state', '')}"
                )
            has_content = True

        # Foreshadowing management (only high-risk items)
        fm = cs.get("foreshadowing_management", [])
        high_risk = [f for f in fm if f.get("risk_level") == "high"]
        if high_risk:
            lines.append("- 即将过期的伏笔：")
            for f in high_risk:
                lines.append(
                    f"  {f.get('description', '?')}"
                    f"（已悬置{f.get('chapters_outstanding', '?')}章，"
                    f"建议{f.get('action_needed', 'maintain')}）"
                )
            has_content = True

        return "\n".join(lines) if has_content else ""
