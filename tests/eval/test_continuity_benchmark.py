"""Pytest integration for continuity benchmark.

Run with:
    pytest tests/eval/test_continuity_benchmark.py -v -s

Set OPENAI_API_KEY, OPENAI_BASE_URL, BUDGET_MODEL env vars.
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
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_continuity_benchmark():
    """Run all built-in benchmark cases and report results."""
    bench = ContinuityBenchmark()
    bench.load_cases(BUILTIN_CASES)

    assert len(bench.cases) == 3, f"Expected 3 built-in cases, got {len(bench.cases)}"

    results = await bench.run()

    assert len(results) == 3
    summary = bench.summary()

    print("\n=== Continuity Benchmark Results ===\n")
    for r in results:
        print(
            f"  {r.case_name:35s}  "
            f"P={r.precision:.2f}  R={r.recall:.2f}  F1={r.f1:.2f}  "
            f"TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives}"
        )
    print(f"\n  Macro: P={summary['macro_precision']:.3f}  "
          f"R={summary['macro_recall']:.3f}  F1={summary['macro_f1']:.3f}\n")

    # Soft assertions — benchmark is informative, not a hard gate
    assert summary["macro_f1"] > 0.0, "F1 should be measurable"
    assert summary["total_bugs_injected"] == 4


@pytest.mark.asyncio
async def test_benchmark_case_loading():
    """Benchmark cases load correctly from JSON."""
    bench = ContinuityBenchmark()
    bench.load_cases(BUILTIN_CASES)

    assert len(bench.cases) == 3
    assert bench.cases[0].name == "character_name_swap"
    assert len(bench.cases[0].injected_bugs) == 2
    assert bench.cases[0].injected_bugs[0].severity == "critical"
