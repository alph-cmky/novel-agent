"""Writer Agent — generates chapter content from outline and context."""

from collections.abc import AsyncIterator

from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.parser import strip_none
from novel_agent.tools.search import SearchContextTool

WRITER_SYSTEM_PROMPT = """你是长篇小说章节执行器。

你的首要目标是保持长篇叙事可靠推进，而不是堆砌辞藻：
1. 不违反 Canon 事实、角色状态和时间线。
2. 完成本章任务，推进至少一个剧情或角色状态。
3. 保持角色动机、知识和能力变化一致。
4. 让章节结尾形成自然的下一步状态，不强行制造反转。

## 指令优先级

1. Canon / 已批准事实
2. 本章必须发生和禁止发生事项
3. 角色当前状态与动机
4. 时间线、因果和未解决剧情线程
5. 叙事风格
6. 通用修辞建议

如果规则冲突，遵守优先级更高的规则；不要擅自新增会改变 Canon 的重大事实。

## 写作要求

- 用具体行动、感官和对话呈现，不用空泛总结代替场景。
- 对话比例服从场景目标，不设全局固定比例。
- 不重复已经完成的剧情推进，不让已确认死亡或离场的角色无理由出现。
- 不强制每章反转或 cliffhanger，但必须留下可继续的叙事状态。
- 只输出章节正文，不输出分析、评分、解释、标题或元信息。
- 充分展开本章关键场景，目标篇幅以本章任务为准。

## 段落原则

- 自然段以叙事单元而非单句为边界。
- 连续动作、感知、环境与人物反应尽量组织在同一自然段。
- 短句不等于短段。
- 不要为了强调频繁制造单句段。
- 对白自然独立成段。
"""


class WriterAgent(BaseAgent):
    name = "writer"

    _DIMENSION_LABELS = {
        "rhythm": "节奏",
        "ai_flavor": "AI味",
        "dialogue": "对话",
        "logic": "逻辑",
        "writing": "文笔",
    }

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
        context_packet: dict | None = None,
        target_chapter_words: int = 0,
        improvement_plan: dict | None = None,
        evolution_version: int = 0,
        orchestrator_strategy: dict | None = None,
    ) -> tuple[str, TraceStep]:
        """Generate a chapter.

        Args:
            chapter_number: Current chapter number.
            outline: Chapter outline / plot points.
            context_packet: Projected context from ContextCompiler.for_writer().
            target_chapter_words: Override per-chapter word count (0 = use default).
            improvement_plan: Structured plan from evolution or human feedback.
            evolution_version: Current evolution version (for prompt wording).
            orchestrator_strategy: Narrative strategy from Orchestrator (stage, pacing, etc.).
        """
        packet = context_packet or {}
        character_context = packet.get("character_context", "")
        world_context = packet.get("world_context", "")
        recent_summary = packet.get("recent_summary", "")
        unresolved_foreshadowings = packet.get("unresolved_foreshadowings", [])
        timeline_events = packet.get("timeline_events", [])
        timeline_findings = packet.get("timeline_findings", [])
        messages = [{"role": "system", "content": self.system_prompt}]

        # Assemble context
        context_parts = [f"## 第{chapter_number}章大纲\n{outline}"]
        plan_text = self._format_improvement_plan(improvement_plan, evolution_version)
        if plan_text:
            context_parts.insert(0, plan_text)
        if orchestrator_strategy:
            strategy_text = self._format_strategy(orchestrator_strategy)
            if strategy_text:
                context_parts.insert(1 if plan_text else 0, strategy_text)
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
        if timeline_events:
            context_parts.append(f"## 已发生的关键事件\n{timeline_events[-10:]}")
        if timeline_findings:
            context_parts.append(f"## 时间线警告\n{timeline_findings[:10]}")

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
        context_packet: dict | None = None,
        target_chapter_words: int = 0,
        improvement_plan: dict | None = None,
        evolution_version: int = 0,
        orchestrator_strategy: dict | None = None,
    ) -> AsyncIterator[str]:
        """Generate a chapter with streaming output. Yields text chunks.

        Unlike write(), this skips tool calling and streams the LLM response directly.
        """
        packet = context_packet or {}
        character_context = packet.get("character_context", "")
        world_context = packet.get("world_context", "")
        recent_summary = packet.get("recent_summary", "")
        unresolved_foreshadowings = packet.get("unresolved_foreshadowings", [])
        timeline_events = packet.get("timeline_events", [])
        timeline_findings = packet.get("timeline_findings", [])
        context_parts = [f"## 第{chapter_number}章大纲\n{outline}"]
        plan_text = self._format_improvement_plan(improvement_plan, evolution_version)
        if plan_text:
            context_parts.insert(0, plan_text)
        if orchestrator_strategy:
            strategy_text = self._format_strategy(orchestrator_strategy)
            if strategy_text:
                context_parts.insert(1 if plan_text else 0, strategy_text)
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
        if timeline_events:
            context_parts.append(f"## 已发生的关键事件\n{timeline_events[-10:]}")
        if timeline_findings:
            context_parts.append(f"## 时间线警告\n{timeline_findings[:10]}")

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

    async def narrative_extension(
        self,
        *,
        current_content: str,
        chapter_number: int,
        chapter_outline: str,
        context_packet: dict | None = None,
        gap_words: int = 500,
    ) -> str:
        """Generate incremental content to extend a short chapter.

        Only produces the continuation — caller appends to existing content.
        Uses minimal context (ending + outline + character state), not full
        chapter context.
        """
        packet = context_packet or {}
        character_context = packet.get("character_context", "")
        unresolved_foreshadowings = packet.get("unresolved_foreshadowings", [])
        ending = current_content[-800:] if len(current_content) > 800 else current_content

        context_parts = [
            f"## 本章大纲\n{chapter_outline}",
            f"## 当前正文结尾\n{ending}",
            f"## 需要续写约 {gap_words} 字",
        ]
        if character_context:
            context_parts.append(f"## 当前角色\n{character_context}")
        if unresolved_foreshadowings:
            context_parts.append(
                "## 待回收伏笔\n" + "\n".join(f"- {item}" for item in unresolved_foreshadowings[:5])
            )

        user_prompt = (
            "上面的正文尚未完成本章目标。请从当前结尾处自然续写，继续推进剧情。\n"
            "续写要求：\n"
            "- 从当前故事的最后状态自然向前发展，不要回头重写已有内容。\n"
            "- 优先发展：当前事件的自然后果 → 人物反应 → 新信息/新发现 → 新决策 → 下一 Beat。\n"
            "- 如果当前 Scene 已经自然收束，优先开启下一个自然 Beat / Scene，"
            "而不是反复扩写当前 Scene。\n"
            "- 不要重复已有的环境描写、神态描写或对白。\n"
            "- 不要总结或解释，直接输出续写正文。\n\n" + "\n\n".join(context_parts)
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content, _ = await self.run_with_tools(
            messages,
            max_rounds=1,
            action=f"extend_ch{chapter_number}",
        )
        return content

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
                "intro": "开篇",
                "development": "发展",
                "climax": "高潮",
                "resolution": "收尾",
                "unit_arc": "单元剧",
                "mini_climax": "小高潮",
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
            lines.append(f"### 建议字数: {cs['suggested_chapter_words']}字")
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
                lines.append(f"  跨单元线索：{'；'.join(ua.get('carry_over_elements', []))}")
            has_content = True

        # pov_config: only multi_perspective / ensemble mode
        if mode in ("multi_perspective", "ensemble") and cs.get("pov_config"):
            persp = cs["pov_config"]
            if persp.get("current_pov"):
                lines.append(
                    f"- 当前POV：{persp['current_pov']}（{persp.get('access_level', 'deep')}）"
                )
            if persp.get("knowledge_gap"):
                lines.append(f"- 视角信息差：{persp['knowledge_gap']}")
            has_content = True

        # time_structure: only when non-linear
        ts = cs.get("time_structure") or {}
        if ts.get("mode") and ts["mode"] != "linear":
            lines.append(
                f"- 时间结构：{ts['mode']}，当前时间线：{ts.get('current_timeline', 'present')}"
            )
            if ts.get("flashback_trigger"):
                lines.append(f"  倒叙触发：{ts['flashback_trigger']}")
            has_content = True

        # stage_boundary: only when boundary detected
        if cs.get("stage_boundary", {}).get("is_boundary"):
            sb = cs["stage_boundary"]
            lines.append(f"- 阶段边界：{sb.get('boundary_type', '?')}")
            lines.append(f"  前阶段持续了{sb.get('previous_stage_duration', '?')}章")
            has_content = True

        # ending_tone: near resolution / ending
        if cs.get("ending_tone"):
            et = cs["ending_tone"]
            lines.append(
                f"- 结尾风格：{et.get('type', '?')}（歧义度:{et.get('ambiguity_level', 'medium')}）"
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
            lines.append(f"- 建议主类型：{sc.get('primary_scene_type', '?')}")
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
                lines.append(f"  {arc.get('character_name', '?')}: {arc.get('current_state', '')}")
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

    @classmethod
    def _format_improvement_plan(cls, plan: dict | None, version: int) -> str:
        """Format an evolution improvement_plan dict into a prompt section."""
        if not plan:
            return ""

        parts = [f"## 进化改进指导 (第 {version + 1} 次迭代)"]
        parts.append(f"这是第 {version + 1} 次改进。前面几轮的改进已经提升了部分维度的质量。")

        focus = plan.get("focus_dimensions", [])
        if focus:
            focus_cn = ", ".join(cls._DIMENSION_LABELS.get(d, d) for d in focus)
            parts.append(f"\n### 本轮重点维度\n{focus_cn}")

        primary = plan.get("primary_instruction", "")
        if primary:
            parts.append(f"\n### 核心指令\n{primary}")

        secondary = plan.get("secondary_instructions", [])
        if secondary:
            parts.append("\n### 辅助指令")
            for s in secondary:
                parts.append(f"- {s}")

        constraints = plan.get("constraints", {})
        preserve = constraints.get("preserve", [])
        if preserve:
            preserve_cn = ", ".join(cls._DIMENSION_LABELS.get(d, d) for d in preserve)
            parts.append(f"\n### 请保持\n{preserve_cn} 方面的已有进步，不要牺牲它们")

        avoid = constraints.get("avoid", [])
        if avoid:
            parts.append("\n### 明确禁止")
            for a in avoid:
                parts.append(f"- {a}")

        # 篇幅与结构完整性保护（防止迭代时字数缩水或仅输出片段）
        parts.append(
            "\n### ⚠️ 篇幅与结构硬性要求（必须严格遵守）\n"
            "1. **输出完整的全章节正文**：以当前版本为基础修改，"
            "绝对禁止只输出修改片段、大纲提要或省略号。\n"
            "2. **只修改必要部分**：聚焦本轮重点维度与核心指令，"
            "未涉及的部分保持上一版的剧情与文字稳定，不要为改而改。\n"
            "3. **字数不得缩水**：保持全章篇幅完整展开，情节充分铺陈，"
            "严禁过度压缩概括。\n"
            "4. **保持连续性**：保留全部核心剧情、人物对话与冲突高潮。"
        )

        return "\n".join(parts)
