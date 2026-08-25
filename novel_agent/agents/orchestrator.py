"""Orchestrator Agent — narrative analysis and pipeline context assembly.

Before each chapter, it analyzes the current narrative position and decides
what the chapter needs. Routing decisions are handled by the LangGraph
conditional edges in graph/chapter.py.
"""

from novel_agent.agents.base import AgentConfig, BaseAgent
from novel_agent.memory.compressor import ContextCompressor
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
根据项目的 narrative_mode 调整你的输出：
- **linear（线性）**：按单一主线推进，输出 storylines 中 role=primary 的一条
- **unit_arc（单元剧）**：标记 unit_arc 字段，主线在单元之间松散串联
- **hybrid（混合）**：单元剧情 + 主线推进，同时输出 storylines 和 unit_arc
- **multi_perspective（多视角）**：标记 pov_config 字段，指定每章的 POV 角色
- **ensemble（群像）**：多线并行，pov_config 标记聚焦角色，storylines 管理各线进度

### 4. 章节策略决策
根据阶段分析和篇幅，决定本章需要：
- 推进哪条故事线（用 storylines 数组替代旧的 primary_storyline 单值）
- **细粒度场景分镜拆解（key_scenes）**：必须将本章拆解为 3-4 个具体承前启后的分镜场景（如：场景一·密室交锋、场景二·追踪暗号、场景三·反转背叛），明确每个场景的发生地点、冲突核心与情绪落点
- 需要什么节奏（快节奏战斗/慢节奏情感/信息揭露）
- 应该在高点还是悬念处结束（ending_type 支持 10 种结局类型）
- 需要回收哪些伏笔
- 高潮阶段内部的位置（climax_sequence，长篇>100章时启用）
- 阶段边界检测（stage_boundary，每10章或世界切换时评估）
- 时间结构标记（time_structure，flashback/parallel/non_linear时启用）
- 章节级视角执行（pov_config，multi_perspective/ensemble时启用）
- 结尾风格（ending_tone，接近全书/大卷结尾时启用）

## 输出格式

根据 narrative_mode 在 chapter_strategy 中补充对应字段。
所有新增字段为可选（可省略或设为null），旧字段保持必填以向后兼容。

```json
{
  "narrative_stage": "intro|development|climax|resolution|unit_arc|mini_climax|transition",
  "stage_analysis": "当前阶段的简短分析",
  "chapter_strategy": {
    "primary_storyline": "主要故事线（向后兼容，推荐同时输出storylines）",
    "storylines": [
      {
        "id": "故事线ID",
        "name": "故事线名称",
        "role": "primary|secondary|tertiary",
        "progress": "60%",
        "chapter_focus": "high|medium|low|background",
        "key_events": ["重要事件"],
        "status": "active|paused|resolved"
      }
    ],
    "pacing": "slow|normal|fast",
    "key_scenes": ["本章必须包含的场景"],
    "ending_type": "cliffhanger|emotional_beat|revelation|...等（见上文10种类型）",
    "foreshadowings_to_address": ["需要回收或强化的伏笔"],
    "suggested_chapter_words": 3000,
    "climax_sequence": null,
    "stage_boundary": null,
    "unit_arc": null,
    "pov_config": null,
    "time_structure": {"mode": "linear", "current_timeline": "present"},
    "ending_tone": null,
    "storyline_intersection": null,
    "character_arcs": [],
    "character_emotional_state": {},
    "tension_profile": null,
    "foreshadowing_management": [],
    "scene_composition": null
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

### 新增字段说明

**climax_sequence**（climax阶段+长篇时输出）：
- current_mini_climax: 当前是第几个小高潮
- total_mini_climaxes: climax阶段共几次起伏
- mini_climax_type: political_turmoil|war|emotional_breakdown|revelation等
- previous_resolution: 上一个mini-climax的解决状态

**stage_boundary**（每10章或世界切换时评估）：
- is_boundary: 是否阶段边界
- boundary_type: world_switch|time_jump|conflict_escalation|resolution_start
- previous_stage_duration: 前阶段持续章数
- estimated_next_stage_duration: 预估下一阶段章数

**unit_arc**（unit_arc/hybrid模式时输出）：
- unit_number, unit_title, unit_type（case_of_the_week|training_arc|filler|character_spotlight）
- mainline_progress, unit_resolution（resolved|unresolved|cliffhanger）
- carry_over_elements: 跨单元线索列表

**pov_config**（multi_perspective/ensemble时输出）：
- current_pov: POV角色名
- pov_shift: 是否视角切换
- access_level: surface|moderate|deep
- knowledge_gap: 该角色不知道的关键信息

**time_structure**（始终输出）：
- mode: linear|flashback|parallel|non_linear
- current_timeline: present|past|future|timeline_a|timeline_b
- flashback_trigger, time_gap, synchronization_points

**ending_tone**（接近全书结尾时输出）：
- type: open_ending等
- ambiguity_level: high|medium|low
- reader_satisfaction, setup_for_next

**storyline_intersection**（多线交汇时输出）：
- has_intersection: 是否交汇
- intersection_type: crossover|parallel|contrast
- intersection_description: 交汇描述

**character_arcs**（每10章评估一次，角色重大事件时立即更新）：
- character_name: 角色名
- arc_stage: growth|fall|redemption|flat|corruption
- arc_progress: 弧线完成百分比
- current_state: 当前心理/性格状态描述
- recent_milestone: 最近一次角色转折事件
- next_milestone: 预计下一次转折事件

**character_emotional_state**（每章输出，标记出场角色的情绪状态）：
- 键为角色名，值为 {mood, trigger, intensity}
- mood: conflicted|hopeful|despairing|angry|detached|fearful|determined|joyful|grieving
- trigger: 导致该情绪的事件
- intensity: high|medium|low

**tension_profile**（每章输出，管理读者情绪体验）：
- chapter_tension: 1-10 本章紧张度
- overall_trend: rising|falling|holding|peak|valley
- recent_chapters_tension: [最近4章的紧张度]
- emotional_tone: 本章情绪基调
- variety_check: {suggestion: 连续同类型过多时的提醒}

**foreshadowing_management**（追踪所有悬而未决的伏笔）：
- id, description, introduced_chapter, estimated_reveal_chapter
- chapters_outstanding: 已悬置章数
- risk_level: low|medium|high（接近泄密阈值时升高）
- action_needed: tease|advance|resolve|maintain
- reader_knows: 读者是否已知
- characters_aware: 知道此伏笔的角色
- characters_unaware: 不知道的角色（制造戏剧性反讽）

**scene_composition**（标记场景类型，避免连续同类型疲劳）：
- primary_scene_type: action|dialogue|introspection|description|transition|mixed
- scene_breakdown: {action, dialogue, introspection, description} 百分比
- recent_dominance: 最近5章主要类型分布
- diversity_warning: 某类型占比过高时的提醒

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
        narrative_mode: str | None = None,
        narrative_perspective: str = "",
        arc_summary: str = "",
        unresolved_foreshadowings: list[str] | None = None,
        context_packet: dict | None = None,
        timeline_events: list[dict] | None = None,
        timeline_findings: list[dict] | None = None,
    ) -> dict:
        """Analyze narrative position and decide chapter strategy.

        Returns a dict with narrative_stage, chapter_strategy, context_needed.
        """
        total_chapters = len(previous_chapters)

        recent = previous_chapters[-3:] if len(previous_chapters) > 3 else previous_chapters
        recent_titles = ", ".join(f"第{c.get('chapter_number', '?')}章" for c in recent)

        length_label = {
            "short": "短篇",
            "medium": "中篇",
            "long": "长篇",
        }.get(story_length, story_length or "长篇")
        if context_packet:
            character_context = context_packet.get("character_context", character_context)
            world_context = context_packet.get("world_context", world_context)
            unresolved_foreshadowings = context_packet.get(
                "unresolved_foreshadowings", unresolved_foreshadowings
            )
            timeline_events = context_packet.get("timeline_events", timeline_events)
            timeline_findings = context_packet.get("timeline_findings", timeline_findings)
        unresolved = unresolved_foreshadowings or []
        foreshadowing_context = "\n".join(f"- {item}" for item in unresolved)
        timeline_context = "\n".join(str(item) for item in (timeline_events or [])[-10:])
        timeline_warnings = "\n".join(str(item) for item in (timeline_findings or [])[:10])

        mode_instruction = self._build_mode_instruction(narrative_mode)
        persp_hint = self._build_perspective_hint(narrative_perspective)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"请分析当前叙事状态并制定第{chapter_number}章的策略。\n\n"
                    f"{mode_instruction}\n"
                    f"{persp_hint}"
                    f"## 篇幅信息\n"
                    f"- 篇幅：{length_label}\n"
                    f"- 目标每章字数：{target_chapter_words}字\n"
                    f"- 已完成章节数：{total_chapters}章\n\n"
                    f"## 本章大纲\n{chapter_outline}\n\n"
                    f"## 已有角色\n{character_context or '暂无'}\n\n"
                    f"## 世界观设定\n{world_context or '暂无'}\n\n"
                    f"## 已有章节\n{recent_titles or '无'}\n\n"
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
                    "primary_storyline": "",
                    "storylines": [],
                    "pacing": "normal",
                    "key_scenes": [],
                    "ending_type": "cliffhanger",
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

        # Track in story arc
        self._story_arc.append(
            {
                "chapter": chapter_number,
                "stage": result.get("narrative_stage", "?"),
                "strategy": result.get("chapter_strategy", {}),
            }
        )

        return result

    @staticmethod
    def _build_mode_instruction(mode: str | None) -> str:
        """Build mode-specific output instructions for the Orchestrator prompt.

        Injects the relevant JSON schema fragments and output requirements
        based on the project's narrative_mode. Returns empty string for
        legacy projects (mode=None).
        """
        if not mode:
            return ""

        base = f"当前叙事模式：{mode}\n\n"

        if mode == "unit_arc":
            return base + (
                "请在 chapter_strategy 中额外输出 unit_arc 字段：\n"
                '  "unit_arc": {\n'
                '    "unit_number": <int>,\n'
                '    "unit_title": "<string>",\n'
                '    "unit_type": "case_of_the_week|training_arc|filler|character_spotlight",\n'
                '    "mainline_progress": "<百分比>",\n'
                '    "unit_resolution": "resolved|unresolved|cliffhanger",\n'
                '    "carry_over_elements": ["<跨单元线索>"]\n'
                "  }\n"
                "同时输出完整的 storylines、time_structure、ending_tone、"
                "tension_profile 和 scene_composition 字段。\n"
            )

        if mode == "hybrid":
            return base + (
                "hybrid 模式需要同时输出 unit_arc 和 storylines：\n"
                "请在 chapter_strategy 中额外输出 unit_arc 字段：\n"
                '  "unit_arc": {\n'
                '    "unit_number": <int>,\n'
                '    "unit_title": "<string>",\n'
                '    "unit_type": "case_of_the_week|training_arc|filler|character_spotlight",\n'
                '    "mainline_progress": "<百分比>",\n'
                '    "unit_resolution": "resolved|unresolved|cliffhanger",\n'
                '    "carry_over_elements": ["<跨单元线索>"]\n'
                "  }\n"
                "同时输出完整的 storylines、time_structure、ending_tone、"
                "storyline_intersection（多线交汇时）、"
                "character_arcs（有里程碑事件时）、character_emotional_state、"
                "tension_profile、foreshadowing_management 和 scene_composition 字段。\n"
            )

        if mode in ("multi_perspective", "ensemble"):
            return base + (
                "请在 chapter_strategy 中额外输出 pov_config 字段：\n"
                '  "pov_config": {\n'
                '    "current_pov": "<角色名>",\n'
                '    "pov_shift": <bool>,\n'
                '    "access_level": "surface|moderate|deep",\n'
                '    "knowledge_gap": "<该角色不知道的关键信息>"\n'
                "  }\n"
                "同时输出完整的 storylines、time_structure、ending_tone、"
                "character_arcs（有里程碑事件时）、character_emotional_state、"
                "tension_profile、foreshadowing_management 和 scene_composition 字段。\n"
            )

        # linear and other modes: always output all fields
        return base + (
            "本章请输出完整的 storylines、time_structure、ending_tone、"
            "character_arcs（有里程碑事件时）、character_emotional_state、"
            "tension_profile、foreshadowing_management 和 scene_composition 字段。\n"
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

    def get_arc_summary(self) -> list[dict]:
        """Return the story arc tracking data."""
        return self._story_arc
