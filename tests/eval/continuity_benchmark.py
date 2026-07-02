"""Continuity Agent benchmark — injects known bugs and scores detection.

Each test case is a chapter pair with injected inconsistencies.
The ContinuityAgent audits the second chapter and we measure:
- precision: detected issues that match ground truth / total detected
- recall: detected issues that match ground truth / total ground truth
- f1: harmonic mean of precision and recall
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectedBug:
    """A known inconsistency injected into a test chapter."""

    category: str  # character, timeline, worldbuilding
    severity: str  # critical, major, minor
    description: str
    location_hint: str  # where in the chapter the bug was placed


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    name: str
    description: str
    chapter_number: int
    chapter_outline: str
    draft_content: str
    injected_bugs: list[InjectedBug] = field(default_factory=list)
    previous_context: str = ""  # context from earlier chapters


@dataclass
class BenchmarkResult:
    """Results for a single benchmark case."""

    case_name: str
    total_injected: int
    total_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    details: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.total_detected > 0:
            self.precision = self.true_positives / self.total_detected
        if self.total_injected > 0:
            self.recall = self.true_positives / self.total_injected
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)


class ContinuityBenchmark:
    """Runs continuity detection benchmarks against injected bugs."""

    def __init__(self):
        self.cases: list[BenchmarkCase] = []
        self.results: list[BenchmarkResult] = []

    def add_case(self, case: BenchmarkCase):
        self.cases.append(case)

    def load_cases(self, cases):
        """Load benchmark cases from a list, JSON string, or file path."""
        if isinstance(cases, list):
            data = cases
        elif os.path.isfile(cases):
            data = json.loads(open(cases).read())
        else:
            data = json.loads(cases)

        for entry in data:
            bugs = [
                InjectedBug(
                    category=b.get("category", ""),
                    severity=b.get("severity", "minor"),
                    description=b.get("description", ""),
                    location_hint=b.get("location_hint", ""),
                )
                for b in entry.get("injected_bugs", [])
            ]
            self.add_case(BenchmarkCase(
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                chapter_number=entry.get("chapter_number", 1),
                chapter_outline=entry.get("chapter_outline", ""),
                draft_content=entry.get("draft_content", ""),
                injected_bugs=bugs,
                previous_context=entry.get("previous_context", ""),
            ))

    def score_detection(self, case: BenchmarkCase, audit_report: dict) -> BenchmarkResult:
        """Compare audit results against ground truth injected bugs."""
        reported = audit_report.get("inconsistencies", [])
        total_injected = len(case.injected_bugs)
        total_detected = len(reported)

        # Simple keyword-overlap matching between reported issues and injected bugs
        matched_bug_indices: set[int] = set()
        details = []

        for issue in reported:
            desc = issue.get("description", "")
            best_match = -1
            best_overlap = 0

            for i, bug in enumerate(case.injected_bugs):
                if i in matched_bug_indices:
                    continue
                overlap = _keyword_overlap(desc, bug.description)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i

            is_match = best_overlap > 0.4 and best_match >= 0
            if is_match and best_match >= 0:
                matched_bug_indices.add(best_match)

            details.append({
                "reported": desc[:100],
                "matched": is_match,
                "best_ground_truth": case.injected_bugs[best_match].description[:100]
                if best_match >= 0 else "",
                "overlap_score": round(best_overlap, 2),
            })

        true_positives = len(matched_bug_indices)
        false_positives = total_detected - true_positives
        false_negatives = total_injected - true_positives

        return BenchmarkResult(
            case_name=case.name,
            total_injected=total_injected,
            total_detected=total_detected,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            details=details,
        )

    async def run(self, persist_dir: str = "./novel-data/chroma_data") -> list[BenchmarkResult]:
        """Run all benchmark cases against the ContinuityAgent."""
        from novel_agent.agents.base import AgentConfig
        from novel_agent.agents.continuity import ContinuityAgent
        from novel_agent.memory.embeddings import ChapterStore

        self.results = []

        for case in self.cases:
            config = AgentConfig(
                model=os.getenv("BUDGET_MODEL", "deepseek-chat"),
                temperature=0.1,
            )
            store = ChapterStore(persist_dir)
            agent = ContinuityAgent(config=config, chapter_store=store, project_id="benchmark")

            report, _ = await agent.audit(
                chapter_number=case.chapter_number,
                draft_content=case.draft_content,
            )

            result = self.score_detection(case, report)
            self.results.append(result)

        return self.results

    def summary(self) -> dict[str, Any]:
        """Aggregate results across all cases."""
        if not self.results:
            return {"error": "No results"}

        total_injected = sum(r.total_injected for r in self.results)
        total_detected = sum(r.total_detected for r in self.results)
        total_tp = sum(r.true_positives for r in self.results)

        macro_precision = sum(r.precision for r in self.results) / len(self.results)
        macro_recall = sum(r.recall for r in self.results) / len(self.results)
        macro_f1 = sum(r.f1 for r in self.results) / len(self.results)

        return {
            "cases": len(self.results),
            "total_bugs_injected": total_injected,
            "total_issues_reported": total_detected,
            "true_positives": total_tp,
            "macro_precision": round(macro_precision, 3),
            "macro_recall": round(macro_recall, 3),
            "macro_f1": round(macro_f1, 3),
            "per_case": [
                {
                    "name": r.case_name,
                    "precision": round(r.precision, 3),
                    "recall": round(r.recall, 3),
                    "f1": round(r.f1, 3),
                    "tp": r.true_positives,
                    "fp": r.false_positives,
                    "fn": r.false_negatives,
                }
                for r in self.results
            ],
        }


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple Jaccard-like keyword overlap between two strings."""
    def tokenize(s: str) -> set[str]:
        keywords = set()
        for phrase in s.replace("，", ",").replace("。", ",").split(","):
            phrase = phrase.strip().lower()
            if len(phrase) >= 2:
                keywords.add(phrase)
        return keywords

    a_tokens = tokenize(text_a)
    b_tokens = tokenize(text_b)

    if not a_tokens and not b_tokens:
        return 0.0
    if not a_tokens or not b_tokens:
        return 0.0

    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


# ── Built-in benchmark cases ──────────────────────────

BUILTIN_CASES = [
    {
        "name": "character_name_swap",
        "description": "主角名字从'林风'突然变成'林峰'",
        "chapter_number": 3,
        "chapter_outline": "主角在城中遭遇伏击，展现出新的能力",
        "draft_content": (
            "林峰走进城主府的大门，守卫拦住了他。\n"
            '"我是来找城主的，"林峰平静地说。\n'
            "守卫看了看他，摇头道：城主今日不见客。\n"
            "林峰微微一笑，从怀中取出一枚令牌。守卫脸色大变，连忙让开道路。\n"
            "穿过庭院，林峰看到了坐在大厅中的城主。\n"
            '"你终于来了，"城主抬起头，"我等你很久了，林风。"\n'
            "林峰纠正道：我是林峰。\n"
            "城主愣了一下，随即笑道：是我记错了。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "critical",
                "description": "主角名字从'林风'变成'林峰'，前后不一致",
                "location_hint": "整章使用'林峰'而前文章节使用'林风'",
            },
            {
                "category": "character",
                "severity": "critical",
                "description": "城主称呼主角为'林风'，主角却自称'林峰'，存在身份混淆",
                "location_hint": "对话中",
            },
        ],
        "previous_context": "前两章中，主角的名字是'林风'，来自青云镇的年轻剑客。",
    },
    {
        "name": "timeline_contradiction",
        "description": "时间线矛盾：前文说3天前发生的事，本章说1周前",
        "chapter_number": 5,
        "chapter_outline": "主角在修炼中突破瓶颈",
        "draft_content": (
            "距离那场大战已经过去了一周，但林风的伤还未痊愈。\n"
            "他盘坐在密室中，感受着体内灵力的流动。\n"
            "一周前的那场战斗，让他险些丧命。\n"
            "不过现在，他感觉自己的修为瓶颈终于松动了。"
        ),
        "injected_bugs": [
            {
                "category": "timeline",
                "severity": "major",
                "description": "大战时间从3天前变成1周前，时间线矛盾",
                "location_hint": "章节开头的时间描述",
            },
        ],
        "previous_context": "第4章结尾：大战发生在3天前，林风受伤后一直在养伤。",
    },
    {
        "name": "worldbuilding_rule_violation",
        "description": "世界观规则违反：前文设定灵力只能通过修炼获得，本章出现了丹药增灵",
        "chapter_number": 4,
        "chapter_outline": "主角获得了一枚神奇的丹药",
        "draft_content": (
            "老者从袖中取出一枚丹药，递到林风面前。\n"
            '"服下此丹，你的灵力可瞬间提升一个大境界，"老者说。\n'
            "林风接过丹药，感受着其中澎湃的灵力波动。\n"
            "他毫不犹豫地吞了下去，体内的灵力果然暴涨。"
        ),
        "injected_bugs": [
            {
                "category": "worldbuilding",
                "severity": "critical",
                "description": (
                    "前文明确设定灵力只能通过自身修炼获得，不能借助外力。"
                    "本章出现了可提升灵力的丹药，违反世界观规则。"
                ),
                "location_hint": "丹药增灵的情节",
            },
        ],
        "previous_context": "这个世界中，灵力只能通过自身的艰苦修炼获得，没有任何捷径。这是铁律。",
    },
]
