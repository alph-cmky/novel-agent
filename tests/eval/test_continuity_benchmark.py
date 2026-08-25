"""Pytest integration for continuity benchmark.

Run with:
    pytest tests/eval/test_continuity_benchmark.py -v -s

Set RUN_LLM_TESTS=1 together with OPENAI_API_KEY, OPENAI_BASE_URL and BUDGET_MODEL.
"""

import os

import pytest

from tests.eval.continuity_benchmark import (
    BUILTIN_CASES,
    ContinuityBenchmark,
)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("RUN_LLM_TESTS") or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_LLM_TESTS=1 and OPENAI_API_KEY to run the paid benchmark",
)
async def test_continuity_benchmark():
    """Run all built-in benchmark cases and report results."""
    bench = ContinuityBenchmark()
    bench.load_cases(BUILTIN_CASES)

    assert len(bench.cases) == 12, f"Expected 12 built-in cases, got {len(bench.cases)}"

    results = await bench.run()

    assert len(results) == 12
    summary = bench.summary()

    print("\n=== Continuity Benchmark Results ===\n")
    for r in results:
        print(
            f"  {r.case_name:35s}  "
            f"P={r.precision:.2f}  R={r.recall:.2f}  F1={r.f1:.2f}  "
            f"TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives}"
        )
    print(
        f"\n  Macro: P={summary['macro_precision']:.3f}  "
        f"R={summary['macro_recall']:.3f}  F1={summary['macro_f1']:.3f}\n"
    )

    # Soft assertions — benchmark is informative, not a hard gate
    assert summary["macro_f1"] > 0.0, "F1 should be measurable"
    assert summary["total_bugs_injected"] == 21


@pytest.mark.asyncio
async def test_benchmark_case_loading():
    """Benchmark cases load correctly from JSON."""
    bench = ContinuityBenchmark()
    bench.load_cases(BUILTIN_CASES)

    assert len(bench.cases) == 12
    assert bench.cases[0].name == "character_name_swap"
    assert len(bench.cases[0].injected_bugs) == 2
    assert bench.cases[0].injected_bugs[0].severity == "critical"
    assert sum(len(c.injected_bugs) for c in bench.cases) == 21


def test_builtin_cases_cover_all_category_severity_combos():
    """BUILTIN_CASES 覆盖 3 类(character/timeline/worldbuilding) × 3 级全组合。"""
    combos = {
        (bug["category"], bug["severity"])
        for case in BUILTIN_CASES
        for bug in case["injected_bugs"]
    }
    assert combos == {
        ("character", "critical"),
        ("character", "major"),
        ("character", "minor"),
        ("timeline", "critical"),
        ("timeline", "major"),
        ("timeline", "minor"),
        ("worldbuilding", "critical"),
        ("worldbuilding", "major"),
        ("worldbuilding", "minor"),
    }


def test_quantity_gradient_cases():
    """存在同章注入 3 个与 5 个 bug 的数量梯度 case。"""
    counts = {len(c["injected_bugs"]) for c in BUILTIN_CASES}
    assert 3 in counts, f"缺少单章 3 bug 的数量梯度 case, got counts={counts}"
    assert 5 in counts, f"缺少单章 5 bug 的数量梯度 case, got counts={counts}"


def test_stress_case_tail_truncation():
    """存在 draft_content >4000 字、且全部 bug 位于 >4000 字符处的截断 stress case。"""
    stress_cases = [
        c
        for c in BUILTIN_CASES
        if len(c["draft_content"]) > 4000
        and all(
            bug["keywords"] and min(c["draft_content"].find(kw) for kw in bug["keywords"]) >= 4000
            for bug in c["injected_bugs"]
        )
    ]
    assert stress_cases, "缺少尾部截断 stress case：draft_content>4000 且所有 bug 特征词位置>4000"
