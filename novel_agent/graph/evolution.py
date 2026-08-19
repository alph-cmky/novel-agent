"""Evolution core logic — Delta computation, termination, improvement plans.

Pure business logic. No LLM calls, no I/O. All functions are deterministic
so they can be unit-tested independently.
"""

from dataclasses import dataclass

# ── Config ──────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Configuration for the evolution loop."""
    max_rounds: int = 5
    convergence_threshold: float = 3.0     # |delta| < this → converged
    quality_drop_threshold: float = -10.0   # delta < this → quality regression
    regression_threshold: float = -5.0       # composite drop relative to best
    ceiling_score: int = 90                  # all dims above this → ceiling stop
    plateau_rounds: int = 2                  # consecutive flat rounds → plateau
    editor_weight: float = 0.5
    continuity_weight: float = 0.3
    dimensions_weight: float = 0.2
    min_length_ratio: float = 0.85
    max_new_critical_errors: int = 0
    max_new_major_errors: int = 0
    outline_regression_tolerance: float = 0.0


DEFAULT_EVO_CONFIG = EvolutionConfig()

# All 5 editor dimensions
EDITOR_DIMENSIONS = ("rhythm", "ai_flavor", "dialogue", "logic", "writing")
QUALITY_DIMENSIONS = EDITOR_DIMENSIONS + (
    "character_fidelity",
    "timeline_consistency",
    "worldbuilding_consistency",
    "outline_adherence",
)


# ── Score helpers ───────────────────────────────────────

def continuity_overall(editor_report: dict, continuity_report: dict) -> int:
    """Continuity overall, neutralized to editor when continuity is unavailable.

    Reasoning models occasionally return empty content; `parse_validated` then
    falls back to `overall_score=0`, which the evolution layer would misread as
    "worst possible score" and spuriously trigger a `regressed` termination.
    When continuity is flagged unavailable, substitute the editor score so the
    continuity term in `composite_score` is neutral rather than dragging the
    composite toward 0.
    """
    if continuity_report.get("unavailable"):
        return editor_report.get("overall_score", 0)
    return continuity_report.get("overall_score", 0)


def editor_overall(editor_report: dict, continuity_report: dict) -> int:
    """Editor overall, neutralized to continuity when editor is unavailable.

    Symmetric to `continuity_overall`. When editor is flagged unavailable
    (empty output / parse failure), substitute the continuity score so the
    editor term — the highest-weighted term in `composite_score` — is neutral
    rather than dragging the composite toward 0.
    """
    if editor_report.get("unavailable"):
        return continuity_report.get("overall_score", 0)
    return editor_report.get("overall_score", 0)


def _neutralized_dimensions(editor_report: dict, editor_score: int) -> dict:
    """Editor dimensions, neutralized to the editor score when unavailable.

    When editor is unavailable its `dimensions` are empty, so `dim_deltas`
    would read as a spurious per-dimension crash (0 vs previous). Substitute
    the neutralized editor score so the dimensions term stays inert.
    """
    dims = editor_report.get("dimensions") or {}
    if editor_report.get("unavailable"):
        return {d: editor_score for d in EDITOR_DIMENSIONS}
    return {d: dims.get(d, 0) for d in EDITOR_DIMENSIONS}


def extract_scores(state: dict) -> dict:
    """Extract structured scores from graph state.

    Returns:
        {
            "editor_overall": int,
            "continuity_overall": int,
            "dimensions": {"rhythm": int, "ai_flavor": int, ...},
        }
    """
    editor = state.get("editor_report") or {}
    continuity = state.get("continuity_report") or {}

    editor_score = editor_overall(editor, continuity)

    return {
        "editor_overall": editor_score,
        "continuity_overall": continuity_overall(editor, continuity),
        "dimensions": _neutralized_dimensions(editor, editor_score),
        "editor_unavailable": bool(editor.get("unavailable")),
    }


def composite_score(scores: dict, config: EvolutionConfig | None = None) -> float:
    """Compute composite score from structured scores.

    Formula: editor * w_e + continuity * w_c + dims_avg * w_d
    """
    cfg = config or DEFAULT_EVO_CONFIG
    editor = scores.get("editor_overall", 0)
    continuity = scores.get("continuity_overall", 0)
    dims = scores.get("dimensions", {})
    dims_avg = sum(dims.values()) / max(len(dims), 1)

    return round(
        editor * cfg.editor_weight
        + continuity * cfg.continuity_weight
        + dims_avg * cfg.dimensions_weight,
        1,
    )


def _severity_counts(report: dict | None) -> dict[str, int]:
    counts = {"critical": 0, "major": 0, "minor": 0}
    for item in (report or {}).get("inconsistencies", []):
        if isinstance(item, dict) and item.get("severity") in counts:
            counts[item["severity"]] += 1
    return counts


def build_quality_guard_report(state: dict, best_state: dict | None = None) -> dict:
    """Extract hard-constraint signals without another model call.

    ``outline_coverage`` and ``required_facts_missing`` are optional fields that
    can be supplied by a future structured checker. Missing optional signals are
    neutral rather than treated as failures.
    """
    draft = (state.get("draft_content") or "").strip()
    best_draft = ((best_state or {}).get("draft_content") or "").strip()
    continuity = state.get("continuity_report") or {}
    worldbuilding = state.get("worldbuilding_report") or {}
    current_errors = _severity_counts(continuity)
    current_conflicts = _severity_counts({"inconsistencies": worldbuilding.get("conflicts", [])})
    for severity, count in current_conflicts.items():
        current_errors[severity] += count

    best_errors = _severity_counts((best_state or {}).get("continuity_report"))
    best_worldbuilding = (best_state or {}).get("worldbuilding_report") or {}
    best_conflicts = _severity_counts({"inconsistencies": best_worldbuilding.get("conflicts", [])})
    for severity, count in best_conflicts.items():
        best_errors[severity] += count

    outline_coverage = state.get("outline_coverage")
    if outline_coverage is None:
        outline_coverage = (state.get("editor_report") or {}).get("outline_coverage")
    best_outline_coverage = (best_state or {}).get("outline_coverage")
    if best_outline_coverage is None:
        best_outline_coverage = ((best_state or {}).get("editor_report") or {}).get(
            "outline_coverage"
        )

    return {
        "length": len(draft),
        "best_length": len(best_draft),
        "length_ratio": round(len(draft) / len(best_draft), 4) if best_draft else 1.0,
        "errors": current_errors,
        "best_errors": best_errors,
        "outline_coverage": outline_coverage,
        "best_outline_coverage": best_outline_coverage,
        "required_facts_missing": state.get("required_facts_missing", 0),
        "best_required_facts_missing": (best_state or {}).get("required_facts_missing", 0),
    }


def check_quality_guards(
    state: dict,
    best_state: dict | None = None,
    config: EvolutionConfig | None = None,
) -> dict:
    """Return hard-constraint status for a candidate version."""
    cfg = config or DEFAULT_EVO_CONFIG
    report = build_quality_guard_report(state, best_state)
    violations: list[str] = []
    if report["length_ratio"] < cfg.min_length_ratio:
        violations.append("length_regression")
    for severity, limit in (
        ("critical", cfg.max_new_critical_errors),
        ("major", cfg.max_new_major_errors),
    ):
        increase = report["errors"][severity] - report["best_errors"][severity]
        if increase > limit:
            violations.append(f"{severity}_consistency_regression")
    current_outline = report["outline_coverage"]
    best_outline = report["best_outline_coverage"]
    if (
        current_outline is not None
        and best_outline is not None
        and current_outline < best_outline - cfg.outline_regression_tolerance
    ):
        violations.append("outline_regression")
    if report["required_facts_missing"] > report["best_required_facts_missing"]:
        violations.append("required_facts_regression")
    return {"passed": not violations, "violations": violations, **report}


def is_better_candidate(
    state: dict,
    best_state: dict,
    config: EvolutionConfig | None = None,
) -> tuple[bool, dict]:
    """Compare versions: hard constraints first, then Pareto, then composite."""
    cfg = config or DEFAULT_EVO_CONFIG
    guard = check_quality_guards(state, best_state, cfg)
    if not guard["passed"]:
        return False, guard

    current = extract_scores(state)
    best = extract_scores(best_state)
    current_dims = current.get("dimensions", {})
    best_dims = best.get("dimensions", {})
    comparable = [d for d in EDITOR_DIMENSIONS if d in current_dims and d in best_dims]
    no_worse = all(current_dims[d] >= best_dims[d] for d in comparable)
    strictly_better = any(current_dims[d] > best_dims[d] for d in comparable)
    current_composite = composite_score(current, cfg)
    best_composite = composite_score(best, cfg)
    pareto = no_worse and strictly_better
    accepted = pareto or current_composite > best_composite
    return accepted, {
        **guard,
        "current_composite": current_composite,
        "best_composite": best_composite,
        "pareto_improved": pareto,
    }


# ── Delta computation ───────────────────────────────────

def compute_delta(current: dict, previous: dict) -> dict:
    """Compute per-dimension and per-report score deltas.

    Args:
        current: scores from current round (from extract_scores)
        previous: scores from previous round

    Returns:
        {
            "editor": int,        # editor overall delta
            "continuity": int,    # continuity overall delta
            "composite": float,   # composite delta
            "dimensions": {d: int, ...},  # per-dimension deltas
            "trend": "improving" | "stagnating" | "regressing" | "mixed",
        }
    """
    delta_editor = current["editor_overall"] - previous.get("editor_overall", 0)
    delta_continuity = current["continuity_overall"] - previous.get("continuity_overall", 0)
    delta_composite = composite_score(current) - composite_score(previous)

    curr_dims = current.get("dimensions", {})
    prev_dims = previous.get("dimensions", {})
    dim_deltas = {}
    for d in EDITOR_DIMENSIONS:
        dim_deltas[d] = curr_dims.get(d, 0) - prev_dims.get(d, 0)

    # Trend analysis
    improving = sum(1 for v in dim_deltas.values() if v > 3)
    regressing = sum(1 for v in dim_deltas.values() if v < -3)
    if regressing > improving:
        trend = "regressing"
    elif improving > 0 and regressing == 0:
        trend = "improving"
    elif improving == 0 and regressing == 0:
        trend = "stagnating"
    else:
        trend = "mixed"

    return {
        "editor": delta_editor,
        "continuity": delta_continuity,
        "composite": round(delta_composite, 1),
        "dimensions": dim_deltas,
        "trend": trend,
    }


# ── Termination decision ────────────────────────────────

def decide_termination(
    delta: dict,
    current_scores: dict,
    best_scores: dict,
    history: list[dict],
    config: EvolutionConfig | None = None,
    current_round: int = 0,
    guard_report: dict | None = None,
) -> tuple[str, str]:
    """Decide whether to stop evolution. Returns (termination_reason, detail).

    Termination conditions (checked in priority order):
    1. Hard constraint violation
    2. Single-dimension quality regression: any dimension delta < -10
    3. Editor quality regression: editor overall delta < -10
    4. Composite regression: composite < best_composite - 5
    5. Ceiling: all dimensions > 90
    6. Max rounds: round >= max_rounds
    7. Convergence: all |dimension delta| < threshold
    8. Plateau: consecutive 2 rounds all |delta| < threshold
    """
    cfg = config or DEFAULT_EVO_CONFIG
    threshold = cfg.convergence_threshold

    # 评估不可用（editor 空输出/解析失败）时，crash/regressed/convergence/plateau
    # 都基于 editor 派生信号，会把「评估失败」误读为「真实退化」。此时仅保留
    # max_rounds 防死循环，其余退化类终止一律跳过，让下一轮重跑评估。
    if current_scores.get("editor_unavailable"):
        if current_round >= cfg.max_rounds:
            return ("max_rounds", f"已达最大轮次 {cfg.max_rounds}")
        return ("", "")

    if guard_report and not guard_report.get("passed", True):
        return (
            "hard_constraint_violation",
            ", ".join(guard_report.get("violations", [])),
        )

    curr_dims = current_scores.get("dimensions", {})
    dim_deltas = delta.get("dimensions", {})

    # 1. Single-dimension crash
    for d in EDITOR_DIMENSIONS:
        if dim_deltas.get(d, 0) < cfg.quality_drop_threshold:
            return ("quality_regression", f"维度 {d} 暴跌 {dim_deltas[d]} 分")

    # 2. Editor crash
    if delta.get("editor", 0) < cfg.quality_drop_threshold:
        return ("quality_regression", f"Editor 总分暴跌 {delta['editor']} 分")

    # 3. Composite regression (compare to best, not previous)
    curr_composite = composite_score(current_scores, cfg)
    best_composite = composite_score(best_scores, cfg)
    if curr_composite < best_composite + cfg.regression_threshold:
        return (
            "quality_regression",
            f"综合分 {curr_composite} < 最佳 {best_composite}"
            f" + {cfg.regression_threshold}",
        )

    # 4. Ceiling — 至少经历 1 轮演化且所有维度达到天花板阈值（防止首轮无迭代直接早退）
    if (current_round >= 1
            and all(curr_dims.get(d, 0) > cfg.ceiling_score for d in EDITOR_DIMENSIONS)
            and current_scores.get("editor_overall", 0) > cfg.ceiling_score
            and current_scores.get("continuity_overall", 0) > cfg.ceiling_score):
        return ("ceiling", f"演化迭代达标且所有维度 > {cfg.ceiling_score}，已达天花板")

    # 5. Max rounds
    if current_round >= cfg.max_rounds:
        return ("max_rounds", f"已达最大轮次 {cfg.max_rounds}")

    # 6. Convergence — all dimension |delta| < threshold this round
    if all(abs(dim_deltas.get(d, 0)) < threshold for d in EDITOR_DIMENSIONS):
        return ("converged", f"所有维度变化 |Δ| < {threshold}")

    # 7. Plateau — last N rounds all had |delta| < threshold
    # The v0 baseline entry has delta=None (no predecessor to compare against),
    # so it can't count as a "stagnant round" — filter it out. (Naive
    # r.get("delta", {}) would still return None here, since the key exists.)
    delta_rounds = [r for r in history if r.get("delta") is not None]
    if len(delta_rounds) >= cfg.plateau_rounds:
        recent = delta_rounds[-cfg.plateau_rounds:]
        plateau = all(
            all(abs(r.get("delta", {}).get("dimensions", {}).get(d, 0)) < threshold
                for d in EDITOR_DIMENSIONS)
            for r in recent
        )
        if plateau:
            return ("plateau", f"连续 {cfg.plateau_rounds} 轮停滞")

    return ("", "")


# ── Improvement plan (rule layer) ───────────────────────

def build_improvement_plan_rule(
    current_scores: dict,
    delta: dict | None,
    config: EvolutionConfig | None = None,
) -> dict:
    """Build a structured improvement plan using deterministic rules.

    When delta is None (first round / no history), identifies weakest dimensions.
    When delta is available, focuses on regressed dimensions + persistent weaknesses.

    Returns:
        {
            "focus_dimensions": [str],
            "primary_instruction": str,
            "secondary_instructions": [str],
            "constraints": {"preserve": [str], "avoid": [str], "strategy_override": {}},
        }
    """
    dims = current_scores.get("dimensions", {})

    if delta is None:
        # First round: target weakest dimensions
        sorted_dims = sorted(dims.items(), key=lambda x: x[1])
        focus = [d for d, s in sorted_dims if s < 60][:3]
        if not focus:
            if len(sorted_dims) >= 2:
                focus = [sorted_dims[0][0], sorted_dims[1][0]]
            else:
                focus = [sorted_dims[0][0]]

        primary = f"初稿完成。重点改进维度：{'、'.join(_dim_label(d) for d in focus)}"
        secondary = _dim_suggestions(focus)
        preserve = [d for d in EDITOR_DIMENSIONS if d not in focus and dims.get(d, 0) >= 70]
        avoid = []
    else:
        dim_deltas = delta.get("dimensions", {})

        # Focus on regressed dimensions (delta < -3) + weak but not improving (< 60, delta < 3)
        regressed = [d for d in EDITOR_DIMENSIONS if dim_deltas.get(d, 0) < -3]
        weak = [d for d in EDITOR_DIMENSIONS
                if dims.get(d, 0) < 60 and dim_deltas.get(d, 0) < 3 and d not in regressed]
        focus = (regressed + weak)[:3]

        preserve = [d for d in EDITOR_DIMENSIONS
                    if dim_deltas.get(d, 0) > 3 and d not in focus]

        if focus:
            primary = (
                f"第{len(focus)}轮改进。"
                f"重点改进：{'、'.join(_dim_label(d) for d in focus)}。"
            )
            if regressed:
                reg_labels = '、'.join(_dim_label(d) for d in regressed)
                primary += f"注意：{reg_labels}出现退步，需要纠正。"
        else:
            # All dimensions stable — fine-tune
            focus = [d for d, s in sorted(dims.items(), key=lambda x: x[1])[:2]]
            primary = "大部分维度已稳定，进行微调优化。"

        secondary = _dim_suggestions(focus)
        avoid = _avoid_patterns(focus, dim_deltas)

    return {
        "focus_dimensions": focus,
        "primary_instruction": primary,
        "secondary_instructions": secondary,
        "constraints": {
            "preserve": preserve,
            "avoid": avoid,
            "strategy_override": {},
        },
    }


def _dim_label(dim: str) -> str:
    """Human-readable dimension label."""
    labels = {
        "rhythm": "节奏",
        "ai_flavor": "AI味",
        "dialogue": "对话",
        "logic": "逻辑",
        "writing": "文笔",
    }
    return labels.get(dim, dim)


def _dim_suggestions(focus: list[str]) -> list[str]:
    """Generate concrete suggestions per dimension."""
    suggestions = {
        "rhythm": [
            "调整句子长短交替，避免连续三个同长度句子",
            "检查场景切换频率，关键场景适当放慢节奏",
        ],
        "ai_flavor": [
            "检查并替换AI高频词：此外、不仅如此、至关重要",
            "结尾不要总结升华，停在动作或画面上",
        ],
        "dialogue": [
            "增加角色间的口语化互动，每段对话后跟简短动作描写",
            "确保每个角色有独特的说话风格和口头禅",
        ],
        "logic": [
            "检查情节因果链是否完整，避免逻辑跳跃",
            "确认角色行为动机与性格设定一致",
        ],
        "writing": [
            "减少过度修饰的长句，增加白描和感官细节",
            "用展示而非告知的方式表达情感",
        ],
    }
    result = []
    for d in focus:
        result.extend(suggestions.get(d, []))
    return result[:6]


def _avoid_patterns(focus: list[str], dim_deltas: dict) -> list[str]:
    """Generate patterns to avoid based on whats regressing."""
    avoid = []
    if "ai_flavor" in focus or dim_deltas.get("ai_flavor", 0) < -3:
        avoid.append("以牺牲文笔为代价降低AI味")
    if "writing" in focus or dim_deltas.get("writing", 0) < -3:
        avoid.append("过度使用形容词和比喻")
    if "dialogue" in focus or dim_deltas.get("dialogue", 0) < -3:
        avoid.append("用大段叙述替代对话推进剧情")
    if "rhythm" in focus or dim_deltas.get("rhythm", 0) < -3:
        avoid.append("重复使用简短的疑问句作为节奏工具")
    return avoid
