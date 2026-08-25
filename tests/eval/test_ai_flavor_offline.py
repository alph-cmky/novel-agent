"""Pytest integration for the AI-flavor offline benchmark.

Run with:
    pytest tests/eval/test_ai_flavor_offline.py -v

Pure rule layer — no LLM required. Samples live in
tests/eval/data/ai_flavor_samples.json.
"""

from tests.eval.ai_flavor_offline import AiFlavorOfflineBenchmark, load_samples


def test_ai_scores_below_human():
    ai, human = load_samples("tests/eval/data/ai_flavor_samples.json")
    result = AiFlavorOfflineBenchmark().run(ai, human)
    assert result["mean_ai"] < result["mean_human"]  # AI 文本平均分更低
    assert result["separation"] >= 0.2  # 分布分离度达标
    assert result["true_positive_rate"] >= 0.6  # 命中率
    assert result["false_positive_rate"] <= 0.4  # 误报率
