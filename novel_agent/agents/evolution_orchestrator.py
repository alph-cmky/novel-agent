"""EvolutionOrchestrator Agent — meta-evaluator for the self-evolution loop.

Not a narrative strategist (that's Orchestrator). This is the meta-evaluator:
- Compares versions via Delta analysis
- Judges whether evolution has converged
- Generates structured improvement plans (rule core + LLM enrichment)
"""

from novel_agent.agents.base import AgentConfig, BaseAgent
from novel_agent.schema.parser import parse_json_response

EVOLUTION_ORCHESTRATOR_SYSTEM_PROMPT = """你是一个小说质量元评估师，负责分析章节写作的进化过程。

## 你的职责

1. **分析版本对比数据**：阅读两轮版本之间的分数变化（Delta），理解哪些维度在进步、哪些在退步
2. **生成具体的改进指导**：用自然语言告诉 Writer 具体如何改进，而不是笼统地说"改好一点"
3. **判断是否需要继续进化**：基于 Delta 数据判断质量是否已经收敛

## 改进指导原则

- 每条指导必须具体可执行（"增加角色对话的个性化口头禅" 而非 "改进对话"）
- 明确指出需要保持的优势维度（"节奏维度 +8 且已连续两轮上升，保持现状"）
- 指出退步维度时给出具体原因（不是"文笔退步了"而是"文笔维度从 78 降至 64，是本轮最大跌幅"）
- 你只会收到分数、Delta、违规项和规则层计划，不会看到正文本身——所有分析必须基于这些评估数据。

## 输出格式

输出 JSON（只输出 JSON）：

```json
{
  "analysis": "简短分析（2-3句）：整体趋势、主要进步、主要退步",
  "primary_instruction": "核心改进指令（1-2句，直指要害）",
  "secondary_instructions": ["具体建议1", "具体建议2", "具体建议3"],
  "constraints": {
    "preserve": ["需要保持的优势维度"],
    "avoid": ["不应出现的模式或语气"],
    "strategy_override": {}
  },
  "continue_evolution": true
}
```

只输出JSON。
"""


class EvolutionOrchestratorAgent(BaseAgent):
    """Meta-evaluator for the self-evolution loop.

    Only called when LLM enrichment is needed. The rule-layer (Delta, termination)
    runs deterministically in services/evolution.py before this agent is invoked.
    """

    name = "evolution_orchestrator"

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)

    @property
    def system_prompt(self) -> str:
        return EVOLUTION_ORCHESTRATOR_SYSTEM_PROMPT

    async def enrich_plan(
        self,
        current_version: int,
        current_scores: dict,
        delta: dict | None,
        rule_plan: dict,
        history: list[dict],
        violations: list[str] | None = None,
    ) -> dict | None:
        """Enrich a rule-generated improvement plan with LLM natural language.

        The LLM is a meta-evaluator: it only sees evaluation data (scores,
        delta, guard violations, rule plan) — never the draft, canon, or full
        history. Keeps enrichment prompts small and analysis data-driven.

        Args:
            current_version: Current version number (0-indexed).
            current_scores: From extract_scores().
            delta: From compute_delta(), or None for first round.
            rule_plan: Rule-generated plan from build_improvement_plan_rule().
            history: Previous evolution history entries.
            violations: Quality-guard violations for the current version.

        Returns:
            Enriched plan dict, or None if LLM fails (caller falls back to rule_plan).
        """
        # Build context for the LLM
        dims = current_scores.get("dimensions", {})
        dim_keys = ["rhythm", "ai_flavor", "dialogue", "logic", "writing"]
        dim_summary = ", ".join(f"{self._dim_label(d)}:{dims.get(d, 0)}" for d in dim_keys)
        editor_score = current_scores.get("editor_overall", 0)
        continuity_score = current_scores.get("continuity_overall", 0)

        delta_text = "（首轮，无对比数据）"
        if delta:
            dim_deltas = delta.get("dimensions", {})
            delta_parts = []
            for d in ["rhythm", "ai_flavor", "dialogue", "logic", "writing"]:
                v = dim_deltas.get(d, 0)
                sign = "+" if v > 0 else ""
                delta_parts.append(f"{self._dim_label(d)}{sign}{v}")
            delta_text = ", ".join(delta_parts)
            delta_text += f"\n整体趋势: {delta.get('trend', '?')}"

        history_text = ""
        if history:
            entries = []
            for h in history[-3:]:
                v = h.get("v", "?")
                e = h.get("editor", "?")
                c = h.get("continuity", "?")
                entries.append(f"v{v}: Editor {e}, Continuity {c}")
            history_text = "版本历史：\n" + "\n".join(entries)

        rule_focus = rule_plan.get("focus_dimensions", [])
        rule_focus_text = ", ".join(self._dim_label(d) for d in rule_focus) if rule_focus else "无"

        violations_text = "无"
        if violations:
            violations_text = "\n".join(f"- {v}" for v in violations)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## 当前版本 v{current_version}\n"
                    f"Editor: {editor_score}/100, Continuity: {continuity_score}/100\n"
                    f"各维度: {dim_summary}\n\n"
                    f"## 版本对比\n{delta_text}\n\n"
                    f"{history_text}\n\n"
                    f"## 质量门违规\n{violations_text}\n\n"
                    f"## 规则层分析\n"
                    f"规则层建议聚焦维度：{rule_focus_text}\n"
                    f"规则层核心指令：{rule_plan.get('primary_instruction', '')}\n\n"
                    f"请在规则层分析的基础上，基于以上分数、Delta 和违规数据，"
                    f"生成更具体的自然语言改进指导。"
                    f"保持 focus_dimensions 与规则层一致，只丰富 instructions 和 constraints。"
                ),
            },
        ]

        try:
            content, _ = await self.run_with_tools(
                messages,
                max_rounds=1,
                action=f"enrich_plan_v{current_version}",
            )
            result = parse_json_response(content)
            if result and result.get("primary_instruction"):
                # Merge LLM enrichment into rule plan structure
                rp = rule_plan
                rc = result.get("constraints", {})
                rpc = rp.get("constraints", {})
                return {
                    "focus_dimensions": rp.get("focus_dimensions", []),
                    "primary_instruction": result.get(
                        "primary_instruction", rp.get("primary_instruction", "")
                    ),
                    "secondary_instructions": result.get(
                        "secondary_instructions", rp.get("secondary_instructions", [])
                    ),
                    "constraints": {
                        "preserve": rc.get("preserve", rpc.get("preserve", [])),
                        "avoid": rc.get("avoid", rpc.get("avoid", [])),
                        # strategy_override 必须是 dict（writer_node 会 dict.update）；
                        # LLM 偶发输出 list/str，这里在源头规范化掉。
                        "strategy_override": (
                            rc.get("strategy_override", {})
                            if isinstance(rc.get("strategy_override"), dict)
                            else {}
                        ),
                    },
                }
        except Exception:
            pass

        return None

    @staticmethod
    def _dim_label(dim: str) -> str:
        labels = {
            "rhythm": "节奏",
            "ai_flavor": "AI味",
            "dialogue": "对话",
            "logic": "逻辑",
            "writing": "文笔",
        }
        return labels.get(dim, dim)
