"""Offline benchmark for AI-flavor rule discrimination.

Runs the deterministic detect_ai_flavor rules over curated samples and reports:
- mean_ai / mean_human: average overall_score for each group
- separation: (mean_human - mean_ai) / 100
- true_positive_rate / false_positive_rate: binary classification at score < 60

Pure rule layer — zero LLM cost. Used as a sanity gate before LLM 横评.
"""

from novel_agent.style.analyzer import detect_ai_flavor


def load_samples(path: str) -> tuple[list[str], list[str]]:
    import json

    data = json.load(open(path))
    return data["ai"], data["human"]


class AiFlavorOfflineBenchmark:
    def run(self, ai_samples: list[str], human_samples: list[str]) -> dict:
        ai_scores = [detect_ai_flavor(t)["overall_score"] for t in ai_samples]
        human_scores = [detect_ai_flavor(t)["overall_score"] for t in human_samples]
        mean_ai = sum(ai_scores) / len(ai_scores)
        mean_human = sum(human_scores) / len(human_scores)
        # 分离度：按 60 分阈值做二分类（<60 判为 AI），算 TPR/FPR
        ai_pred = [s < 60 for s in ai_scores]
        human_pred = [s < 60 for s in human_scores]
        tpr = sum(ai_pred) / len(ai_pred)
        fpr = sum(human_pred) / len(human_pred)
        return {
            "mean_ai": round(mean_ai, 1),
            "mean_human": round(mean_human, 1),
            "separation": round((mean_human - mean_ai) / 100, 3),
            "true_positive_rate": round(tpr, 3),
            "false_positive_rate": round(fpr, 3),
            "ai_scores": ai_scores,
            "human_scores": human_scores,
        }
