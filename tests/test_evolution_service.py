"""Tests for the evolution core logic — Delta, termination, improvement plans."""

from novel_agent.schema.enums import EvolutionAction
from novel_agent.services.evolution import (
    EDITOR_DIMENSIONS,
    EvolutionConfig,
    EvolutionService,
    build_improvement_plan_rule,
    check_quality_guards,
    composite_score,
    compute_delta,
    continuity_overall,
    decide_termination,
    editor_overall,
    extract_scores,
    is_better_candidate,
)


class TestExtractScores:
    def test_extracts_basic_scores(self):
        state = {
            "editor_report": {
                "overall_score": 72,
                "dimensions": {
                    "rhythm": 70,
                    "ai_flavor": 65,
                    "dialogue": 75,
                    "logic": 80,
                    "writing": 70,
                },
            },
            "continuity_report": {"overall_score": 80},
        }
        scores = extract_scores(state)
        assert scores["editor_overall"] == 72
        assert scores["continuity_overall"] == 80
        assert scores["dimensions"]["rhythm"] == 70
        assert scores["dimensions"]["ai_flavor"] == 65

    def test_handles_empty_state(self):
        scores = extract_scores({})
        assert scores["editor_overall"] == 0
        assert scores["continuity_overall"] == 0
        assert all(scores["dimensions"][d] == 0 for d in EDITOR_DIMENSIONS)

    def test_handles_missing_dimensions(self):
        state = {
            "editor_report": {"overall_score": 80, "dimensions": {"rhythm": 70}},
            "continuity_report": {"overall_score": 85},
        }
        scores = extract_scores(state)
        assert scores["dimensions"]["rhythm"] == 70
        assert scores["dimensions"]["ai_flavor"] == 0  # missing → 0

    def test_unavailable_continuity_neutralized_to_editor(self):
        state = {
            "editor_report": {"overall_score": 72, "dimensions": {}},
            "continuity_report": {"unavailable": True, "overall_score": 0},
        }
        scores = extract_scores(state)
        assert scores["continuity_overall"] == 72

    def test_unavailable_editor_neutralized_to_continuity(self):
        state = {
            "editor_report": {"unavailable": True, "overall_score": 0, "dimensions": {}},
            "continuity_report": {"overall_score": 75},
        }
        scores = extract_scores(state)
        assert scores["editor_overall"] == 75
        assert scores["editor_unavailable"] is True
        # dimensions 中和到替身分，避免 dim_delta 假暴跌触发 crash
        assert all(scores["dimensions"][d] == 75 for d in EDITOR_DIMENSIONS)


class TestContinuityOverall:
    def test_unavailable_substitutes_editor(self):
        assert (
            continuity_overall({"overall_score": 88}, {"unavailable": True, "overall_score": 0})
            == 88
        )

    def test_available_returns_own_score(self):
        assert continuity_overall({"overall_score": 88}, {"overall_score": 75}) == 75

    def test_missing_reports_default_to_zero(self):
        assert continuity_overall({"overall_score": 88}, {}) == 0


class TestEditorOverall:
    def test_unavailable_substitutes_continuity(self):
        assert (
            editor_overall({"unavailable": True, "overall_score": 0}, {"overall_score": 75}) == 75
        )

    def test_available_returns_own_score(self):
        assert editor_overall({"overall_score": 88}, {"overall_score": 75}) == 88

    def test_missing_reports_default_to_zero(self):
        assert editor_overall({"unavailable": True}, {}) == 0


class TestCompositeScore:
    def test_basic_composite(self):
        scores = {
            "editor_overall": 80,
            "continuity_overall": 90,
            "dimensions": {
                "rhythm": 75,
                "ai_flavor": 70,
                "dialogue": 85,
                "logic": 80,
                "writing": 75,
            },
            "style_structure_score": 77,
        }
        # composite = 80*0.5 + 90*0.3 + 77*0.2 = 40 + 27 + 15.4 = 82.4
        result = composite_score(scores)
        assert result == 82.4

    def test_custom_weights(self):
        cfg = EvolutionConfig(editor_weight=0.6, continuity_weight=0.2, dimensions_weight=0.2)
        scores = {
            "editor_overall": 80,
            "continuity_overall": 90,
            "dimensions": {
                "rhythm": 75,
                "ai_flavor": 70,
                "dialogue": 85,
                "logic": 80,
                "writing": 75,
            },
            "style_structure_score": 77,
        }
        # 80*0.6 + 90*0.2 + 77*0.2 = 48 + 18 + 15.4 = 81.4
        result = composite_score(scores, cfg)
        assert result == 81.4

    def test_empty_dimensions(self):
        scores = {
            "editor_overall": 70,
            "continuity_overall": 75,
            "dimensions": {},
            "style_structure_score": 0,
        }
        result = composite_score(scores)
        # composite = 70*0.5 + 75*0.3 + 0*0.2 = 57.5
        assert result == 57.5

    def test_style_structure_defaults_to_100_when_missing(self):
        """Missing style_structure_score defaults to 100 (neutral)."""
        scores = {
            "editor_overall": 80,
            "continuity_overall": 90,
            "dimensions": {},
        }
        # 80*0.5 + 90*0.3 + 100*0.2 = 40 + 27 + 20 = 87
        result = composite_score(scores)
        assert result == 87.0


class TestComputeDelta:
    def test_all_improving(self):
        current = {
            "editor_overall": 80,
            "continuity_overall": 85,
            "dimensions": {
                "rhythm": 75,
                "ai_flavor": 70,
                "dialogue": 80,
                "logic": 80,
                "writing": 75,
            },
        }
        previous = {
            "editor_overall": 70,
            "continuity_overall": 82,
            "dimensions": {
                "rhythm": 65,
                "ai_flavor": 60,
                "dialogue": 75,
                "logic": 78,
                "writing": 70,
            },
        }
        delta = compute_delta(current, previous)
        assert delta["editor"] == 10
        assert delta["continuity"] == 3
        assert delta["trend"] == "improving"

    def test_mixed(self):
        current = {
            "editor_overall": 75,
            "continuity_overall": 80,
            "dimensions": {
                "rhythm": 70,
                "ai_flavor": 75,
                "dialogue": 60,
                "logic": 80,
                "writing": 65,
            },
        }
        previous = {
            "editor_overall": 72,
            "continuity_overall": 82,
            "dimensions": {
                "rhythm": 68,
                "ai_flavor": 70,
                "dialogue": 68,
                "logic": 82,
                "writing": 68,
            },
        }
        delta = compute_delta(current, previous)
        # ai_flavor: +5 (improving), dialogue: -8 (regressing)
        assert delta["trend"] == "mixed"

    def test_regressing(self):
        current = {
            "editor_overall": 68,
            "continuity_overall": 78,
            "dimensions": {
                "rhythm": 65,
                "ai_flavor": 62,
                "dialogue": 60,
                "logic": 75,
                "writing": 62,
            },
        }
        previous = {
            "editor_overall": 75,
            "continuity_overall": 82,
            "dimensions": {
                "rhythm": 72,
                "ai_flavor": 70,
                "dialogue": 68,
                "logic": 80,
                "writing": 68,
            },
        }
        delta = compute_delta(current, previous)
        # all negative
        assert delta["trend"] == "regressing"


class TestDecideTermination:
    def _scores(self, editor=80, continuity=85, **dims):
        dims_dict = {"rhythm": 75, "ai_flavor": 70, "dialogue": 80, "logic": 80, "writing": 75}
        dims_dict.update(dims)
        return {
            "editor_overall": editor,
            "continuity_overall": continuity,
            "dimensions": dims_dict,
        }

    def _delta(self, editor=0, continuity=0, **dim_deltas):
        dims = {d: 0 for d in EDITOR_DIMENSIONS}
        dims.update(dim_deltas)
        return {
            "editor": editor,
            "continuity": continuity,
            "dimensions": dims,
            "trend": "stagnating",
        }

    def test_no_termination_when_improving(self):
        """With deltas above convergence threshold, should NOT terminate."""
        reason, _ = decide_termination(
            delta=self._delta(editor=5, ai_flavor=5, dialogue=4),
            current_scores=self._scores(editor=80),
            best_scores=self._scores(editor=75),
            history=[],
            current_round=1,
        )
        assert reason == ""

    def test_crash_single_dimension(self):
        """Any dimension delta < -10 → quality regression."""
        reason, detail = decide_termination(
            delta=self._delta(ai_flavor=-15),
            current_scores=self._scores(ai_flavor=55),
            best_scores=self._scores(),
            history=[],
            current_round=1,
        )
        assert reason == "quality_regression"
        assert "ai_flavor" in detail

    def test_max_rounds(self):
        reason, _ = decide_termination(
            delta=self._delta(),
            current_scores=self._scores(),
            best_scores=self._scores(),
            history=[],
            current_round=5,
            config=EvolutionConfig(max_rounds=5),
        )
        assert reason == "max_rounds"

    def test_convergence(self):
        """All |delta| < threshold → converged."""
        reason, _ = decide_termination(
            delta=self._delta(rhythm=1, ai_flavor=2, dialogue=-1, logic=2, writing=-2),
            current_scores=self._scores(),
            best_scores=self._scores(),
            history=[],
            current_round=2,
            config=EvolutionConfig(convergence_threshold=3.0),
        )
        assert reason == "converged"

    def test_ceiling(self):
        """All dimensions > 90 → ceiling."""
        reason, _ = decide_termination(
            delta=self._delta(),
            current_scores=self._scores(
                editor=95,
                continuity=92,
                rhythm=92,
                ai_flavor=93,
                dialogue=91,
                logic=95,
                writing=94,
            ),
            best_scores=self._scores(),
            history=[],
            current_round=1,
        )
        assert reason == "ceiling"

    def test_regression_vs_best(self):
        """Composite drops below best-5 → quality regression."""
        # current composite should be lower than best composite
        reason, _ = decide_termination(
            delta=self._delta(editor=-15),
            current_scores=self._scores(editor=55, continuity=60),
            best_scores=self._scores(editor=80, continuity=85),
            history=[],
            current_round=2,
        )
        assert reason == "quality_regression"

    def test_editor_unavailable_skips_degradation(self):
        """editor 评估不可用时跳过 crash/regressed 终止，仅保留 max_rounds。"""
        scores = self._scores(editor=0, continuity=0)
        scores["editor_unavailable"] = True
        reason, _ = decide_termination(
            delta=self._delta(editor=-80, ai_flavor=-80),
            current_scores=scores,
            best_scores=self._scores(editor=80, continuity=85),
            history=[],
            current_round=2,
        )
        assert reason == ""

    def test_editor_unavailable_still_respects_max_rounds(self):
        """editor 不可用时 max_rounds 仍生效，防死循环。"""
        scores = self._scores(editor=0, continuity=0)
        scores["editor_unavailable"] = True
        reason, _ = decide_termination(
            delta=self._delta(),
            current_scores=scores,
            best_scores=self._scores(),
            history=[],
            current_round=5,
            config=EvolutionConfig(max_rounds=5),
        )
        assert reason == "max_rounds"

    def test_plateau_skips_none_delta_baseline(self):
        """The v0 baseline entry has delta=None — plateau check must skip it.

        Regression: dict.get("delta", {}) returns None (not the default) when the
        key exists with value None, so the plateau branch crashed with
        ``'NoneType' object has no attribute 'get'`` once history reached
        plateau_rounds (2) entries.
        """
        flat = self._delta(rhythm=1, ai_flavor=1, dialogue=1, logic=1, writing=1)
        dims = self._scores()["dimensions"]
        history = [
            {"v": 0, "delta": None, "editor": 80, "continuity": 85, "dimensions": dims},
            {"v": 1, "delta": flat, "editor": 80, "continuity": 85, "dimensions": dims},
            {"v": 2, "delta": flat, "editor": 80, "continuity": 85, "dimensions": dims},
        ]
        # Current delta is NOT flat (rhythm=5) so convergence doesn't short-circuit;
        # this lets the plateau branch actually run over the history.
        reason, _ = decide_termination(
            delta=self._delta(rhythm=5, ai_flavor=1, dialogue=1, logic=1, writing=1),
            current_scores=self._scores(),
            best_scores=self._scores(),
            history=history,
            current_round=3,
        )
        assert reason == "plateau"


class TestBuildImprovementPlan:
    def _scores(self, **dims):
        dims_dict = {"rhythm": 70, "ai_flavor": 65, "dialogue": 75, "logic": 80, "writing": 70}
        dims_dict.update(dims)
        return {"editor_overall": 72, "continuity_overall": 80, "dimensions": dims_dict}

    def test_first_round_targets_weakest(self):
        """No delta → target weakest dimensions."""
        plan = build_improvement_plan_rule(self._scores(), delta=None)
        assert "ai_flavor" in plan["focus_dimensions"]  # weakest at 65
        assert plan["primary_instruction"]  # should have some instruction

    def test_with_delta_focuses_on_regressed(self):
        """Delta with regressed dimensions → focus on them."""
        current = self._scores(rhythm=68, ai_flavor=75)
        delta = {
            "editor": 3,
            "continuity": 2,
            "dimensions": {"rhythm": -5, "ai_flavor": 10, "dialogue": 2, "logic": 2, "writing": 1},
            "trend": "mixed",
        }
        plan = build_improvement_plan_rule(current, delta=delta)
        assert "rhythm" in plan["focus_dimensions"]  # regressed

    def test_preserve_improved_dimensions(self):
        """Dimensions that improved >+3 should be in preserve."""
        current = self._scores(rhythm=68, ai_flavor=80)
        delta = {
            "editor": 5,
            "continuity": 2,
            "dimensions": {"rhythm": -5, "ai_flavor": 10, "dialogue": 2, "logic": 2, "writing": 1},
            "trend": "mixed",
        }
        plan = build_improvement_plan_rule(current, delta=delta)
        assert "ai_flavor" in plan["constraints"]["preserve"]  # improved +10

    def test_style_gate_fail_injects_structure_instructions(self):
        """style_gate=FAIL → secondary_instructions must include fragmentation guidance."""
        scores = self._scores()
        scores["style_gate"] = "FAIL"
        scores["style_structure_score"] = 30
        plan = build_improvement_plan_rule(scores, delta=None)
        assert any("碎片化" in s for s in plan["secondary_instructions"])
        assert any("一句一段" in s for s in plan["constraints"]["avoid"])

    def test_style_gate_warning_injects_soft_guidance(self):
        """style_gate=WARNING → softer structure instruction, no FAIL-level avoid."""
        scores = self._scores()
        scores["style_gate"] = "WARNING"
        scores["style_structure_score"] = 60
        plan = build_improvement_plan_rule(scores, delta=None)
        assert any("偏碎" in s for s in plan["secondary_instructions"])
        assert any("单句独立成段" in s for s in plan["constraints"]["avoid"])

    def test_style_gate_pass_no_style_instructions(self):
        """style_gate=PASS → no style-structure instructions injected."""
        scores = self._scores()
        scores["style_gate"] = "PASS"
        scores["style_structure_score"] = 95
        plan = build_improvement_plan_rule(scores, delta=None)
        assert not any("碎片化" in s for s in plan["secondary_instructions"])
        assert not any("偏碎" in s for s in plan["secondary_instructions"])


def _candidate_state(content: str, *, critical: int = 0, outline: float = 1.0):
    return {
        "draft_content": content,
        "outline_coverage": outline,
        "editor_report": {
            "overall_score": 80,
            "outline_coverage": outline,
            "dimensions": {
                "rhythm": 80,
                "ai_flavor": 80,
                "dialogue": 80,
                "logic": 80,
                "writing": 80,
            },
        },
        "continuity_report": {
            "overall_score": 80,
            "inconsistencies": [{"severity": "critical", "category": "timeline"}] * critical,
        },
    }


def test_quality_guard_rejects_length_regression():
    best = _candidate_state("x" * 100)
    current = _candidate_state("x" * 50)
    report = check_quality_guards(current, best)
    assert report["passed"] is False
    assert "length_regression" in report["violations"]


def test_quality_guard_rejects_new_critical_error():
    best = _candidate_state("x" * 100)
    current = _candidate_state("x" * 100, critical=1)
    report = check_quality_guards(current, best)
    assert report["passed"] is False
    assert "critical_consistency_regression" in report["violations"]


def test_better_candidate_requires_guards_before_composite():
    best = _candidate_state("x" * 100)
    current = _candidate_state("x" * 50)
    accepted, report = is_better_candidate(current, best)
    assert accepted is False
    assert report["passed"] is False


def test_quality_guard_rejects_style_gate_fail():
    """style_gate=FAIL is a hard violation — candidate must not replace a PASS best."""
    best = _candidate_state("x" * 100)
    best["style_report"] = {"style_gate": "PASS", "paragraph_structure_score": 90}

    current = _candidate_state("x" * 100)
    current["style_report"] = {"style_gate": "FAIL", "paragraph_structure_score": 20}

    report = check_quality_guards(current, best)
    assert report["passed"] is False
    assert "style_gate_fail" in report["violations"]


def test_quality_guard_passes_with_style_gate_warning():
    """style_gate=WARNING is not a hard violation — only FAIL blocks."""
    best = _candidate_state("x" * 100)
    best["style_report"] = {"style_gate": "PASS", "paragraph_structure_score": 90}

    current = _candidate_state("x" * 100)
    current["style_report"] = {"style_gate": "WARNING", "paragraph_structure_score": 60}

    report = check_quality_guards(current, best)
    assert report["passed"] is True
    assert "style_gate_fail" not in report["violations"]


def test_is_better_candidate_rejects_style_gate_fail_even_with_higher_score():
    """Even if editor/continuity are higher, style_gate=FAIL blocks replacement."""
    best = _candidate_state("x" * 100)
    best["editor_report"]["overall_score"] = 70
    best["style_report"] = {"style_gate": "PASS", "paragraph_structure_score": 90}

    current = _candidate_state("x" * 100)
    current["editor_report"]["overall_score"] = 95  # higher than best
    current["style_report"] = {"style_gate": "FAIL", "paragraph_structure_score": 20}

    accepted, report = is_better_candidate(current, best)
    assert accepted is False
    assert "style_gate_fail" in report["violations"]


def test_evolution_service_returns_stop_decision_without_changing_reason():
    helper = TestDecideTermination()
    decision = EvolutionService.evaluate(
        delta=helper._delta(),
        current_scores=helper._scores(),
        best_scores=helper._scores(),
        history=[],
        current_round=5,
        config=EvolutionConfig(max_rounds=5),
    )

    assert decision.action is EvolutionAction.STOP
    assert decision.reason == "max_rounds"
    assert "已达最大轮次" in decision.details["detail"]


def test_evolution_service_returns_continue_decision_with_empty_termination():
    helper = TestDecideTermination()
    decision = EvolutionService.evaluate(
        delta=helper._delta(editor=5, ai_flavor=5, dialogue=4),
        current_scores=helper._scores(editor=80),
        best_scores=helper._scores(editor=75),
        history=[],
        current_round=1,
    )

    assert decision.action is EvolutionAction.CONTINUE
    assert decision.reason == ""
