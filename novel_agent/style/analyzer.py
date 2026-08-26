"""Style analysis — deterministic text measurement for Chinese novel writing.

Four dimensions, all 0-LLM:
1. AI Flavor Evidence — banned phrases, clichés, sentence patterns (evidence/warning)
2. Paragraph Structure — fragmentation, short narrative, consecutive shorts
3. Sentence Rhythm — length variety
4. Dialogue Statistics — descriptive, not a quality gate

Core principle: LLM does literary judgment; code does measurement.
"""

import re
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

from novel_agent.config import (
    MAX_CONSECUTIVE_SHORT_PARAGRAPHS,
    SHORT_NARRATIVE_PARAGRAPH_THRESHOLD,
    SHORT_NARRATIVE_RATIO_THRESHOLD,
    SINGLE_SENTENCE_RATIO_THRESHOLD,
)

# ── Models ──────────────────────────────────────────────


class StyleProfile(BaseModel):
    """Optional style preferences reserved for future style-aware analysis."""


class StyleIssue(BaseModel):
    """A detected style issue with extensible evidence fields."""

    model_config = ConfigDict(extra="allow")

    type: str
    severity: str
    count: int = 1


class ParagraphStructureReport(BaseModel):
    """Paragraph-level structural analysis — detects fragmentation patterns."""

    paragraph_count: int = 0
    narrative_paragraph_count: int = 0
    dialogue_paragraph_count: int = 0
    mixed_paragraph_count: int = 0
    median_narrative_paragraph_length: float = 0
    short_narrative_ratio: float = 0
    single_sentence_narrative_ratio: float = 0
    max_consecutive_short_narrative_paragraphs: int = 0
    fragmentation_score: float = Field(ge=0, le=100, default=100)
    issues: list[str] = Field(default_factory=list)


class StyleReport(BaseModel):
    """Normalized style analysis result."""

    ai_flavor_score: float = Field(ge=0, le=100)
    paragraph_structure_score: float = Field(ge=0, le=100)
    sentence_rhythm_score: float = Field(ge=0, le=100)
    dialogue_score: float = Field(ge=0, le=100)
    issues: list[StyleIssue] = Field(default_factory=list)
    paragraph_structure: dict | None = None


# ── Dialogue detection ──────────────────────────────────

_DIALOGUE_RE = re.compile(
    r"\u201c([^\u201d]*?)\u201d"
    r"|\u300c([^\u300d]*?)\u300d"
    r"|\u300e([^\u300f]*?)\u300f"
    r'|"([^"]*?)"'
    r"|\u2018([^\u2019]*?)\u2019"
)


def _extract_dialogue_content(paragraph: str) -> str:
    """Extract all text inside paired quotation marks."""
    parts = _DIALOGUE_RE.findall(paragraph)
    return "".join(next((m for m in match if m), "") for match in parts)


def _classify_paragraph(paragraph: str) -> str:
    """Classify a paragraph as NARRATIVE, DIALOGUE, or MIXED.

    - NARRATIVE: no dialogue content
    - DIALOGUE: non-dialogue text < 30% of paragraph (quotes are punctuation)
    - MIXED: has dialogue but substantial narrative remainder
    """
    dialogue = _extract_dialogue_content(paragraph)
    if not dialogue:
        return "NARRATIVE"
    narrative_remainder = _DIALOGUE_RE.sub("", paragraph).strip()
    para_len = max(len(paragraph), 1)
    narrative_ratio = len(narrative_remainder) / para_len
    if narrative_ratio < 0.3:
        return "DIALOGUE"
    return "MIXED"


def _is_single_sentence(paragraph: str) -> bool:
    """Check if a paragraph contains at most one sentence."""
    sentences = [s.strip() for s in re.split(r"[。！？!?]", paragraph) if s.strip()]
    return len(sentences) <= 1


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs (scene boundaries filtered out)."""
    return [p.strip() for p in text.split("\n") if p.strip()]


# ── Banned phrases ────────────────────────────────────

BANNED_CONNECTORS = [
    "此外",
    "不仅如此",
    "更重要的是",
    "总而言之",
    "综上所述",
    "基于以上分析",
    "值得注意的是",
    "不难发现",
    "毫无疑问",
    "由此可见",
    "换言之",
    "也就是说",
]

BANNED_EMPHASIS = [
    "至关重要",
    "不可忽视",
    "深入探讨",
    "深刻揭示了",
    "具有重要的现实意义",
    "必须指出的是",
]

BANNED_CLICHES = [
    "他的眼中闪过一丝",
    "她的嘴角微微上扬",
    "他的心中涌起一股",
    "一种难以言喻的",
    "心中充满了",
    "眼中闪过一抹",
    "嘴角勾起一抹",
    "内心深处",
]

BANNED_SENTENCE_PATTERNS = [
    (r"(?:这|那)?不是[^，。,\.]{1,30}而是[^，。,\.]{1,30}", "翻案句式「不是…而是…」"),
    (r"——", "破折号（叙事中建议削减）"),
    (r"(?:值得注意|必须承认|实事求是|坦白)地说", "冒号讲义腔"),
]

HOOK_EVIDENCE_PATTERNS = ["突然", "忽然", "就在这时", "紧接着", "就在此时", "下一秒"]

SUMMARY_PATTERNS = [
    "总之",
    "总而言之",
    "通过",
    "这次经历",
    "这天的经历",
    "他明白了",
    "她终于明白",
    "他学到了",
    "她意识到",
]


# ── Structural checks ─────────────────────────────────


class ParagraphStructureAnalyzer:
    """Deterministic paragraph structure analysis — 0 LLM calls."""

    def analyze(self, text: str) -> ParagraphStructureReport:
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return ParagraphStructureReport()

        classifications = [_classify_paragraph(p) for p in paragraphs]

        narrative_indices = [
            i for i, c in enumerate(classifications) if c in ("NARRATIVE", "MIXED")
        ]
        narrative_count = sum(1 for c in classifications if c == "NARRATIVE")
        dialogue_count = sum(1 for c in classifications if c == "DIALOGUE")
        mixed_count = sum(1 for c in classifications if c == "MIXED")

        narrative_lengths = [len(paragraphs[i]) for i in narrative_indices]
        median_len = float(median(narrative_lengths)) if narrative_lengths else 0

        short_threshold = SHORT_NARRATIVE_PARAGRAPH_THRESHOLD
        short_narrative_indices = [
            i for i in narrative_indices if len(paragraphs[i]) < short_threshold
        ]
        short_ratio = len(short_narrative_indices) / max(len(narrative_indices), 1)

        single_sentence_indices = [
            i for i in narrative_indices if _is_single_sentence(paragraphs[i])
        ]
        single_sentence_ratio = len(single_sentence_indices) / max(len(narrative_indices), 1)

        max_consecutive = self._max_consecutive_short(paragraphs, classifications, short_threshold)

        frag_score = self._fragmentation_score(
            short_ratio, single_sentence_ratio, max_consecutive, median_len
        )

        issues: list[str] = []
        if short_ratio > SHORT_NARRATIVE_RATIO_THRESHOLD:
            issues.append(f"短叙述段比例过高: {short_ratio:.0%}")
        if single_sentence_ratio > SINGLE_SENTENCE_RATIO_THRESHOLD:
            issues.append(f"单句叙述段比例过高: {single_sentence_ratio:.0%}")
        if max_consecutive > MAX_CONSECUTIVE_SHORT_PARAGRAPHS:
            issues.append(f"连续短叙述段: {max_consecutive}")

        return ParagraphStructureReport(
            paragraph_count=len(paragraphs),
            narrative_paragraph_count=narrative_count,
            dialogue_paragraph_count=dialogue_count,
            mixed_paragraph_count=mixed_count,
            median_narrative_paragraph_length=median_len,
            short_narrative_ratio=round(short_ratio, 3),
            single_sentence_narrative_ratio=round(single_sentence_ratio, 3),
            max_consecutive_short_narrative_paragraphs=max_consecutive,
            fragmentation_score=frag_score,
            issues=issues,
        )

    @staticmethod
    def _max_consecutive_short(
        paragraphs: list[str],
        classifications: list[str],
        short_threshold: int,
    ) -> int:
        max_run = 0
        current = 0
        for i, cls in enumerate(classifications):
            if cls in ("NARRATIVE", "MIXED") and len(paragraphs[i]) < short_threshold:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    @staticmethod
    def _fragmentation_score(
        short_ratio: float,
        single_ratio: float,
        max_consec: int,
        median_len: float,
    ) -> float:
        """100 = natural, 0 = severely fragmented. Explainable, no complex model."""
        score = 100.0

        if short_ratio > SHORT_NARRATIVE_RATIO_THRESHOLD:
            excess = (short_ratio - SHORT_NARRATIVE_RATIO_THRESHOLD) / max(
                1 - SHORT_NARRATIVE_RATIO_THRESHOLD, 0.01
            )
            score -= min(30, excess * 30)

        if single_ratio > SINGLE_SENTENCE_RATIO_THRESHOLD:
            excess = (single_ratio - SINGLE_SENTENCE_RATIO_THRESHOLD) / max(
                1 - SINGLE_SENTENCE_RATIO_THRESHOLD, 0.01
            )
            score -= min(25, excess * 25)

        if max_consec > MAX_CONSECUTIVE_SHORT_PARAGRAPHS:
            score -= min(25, (max_consec - MAX_CONSECUTIVE_SHORT_PARAGRAPHS) * 8)

        if median_len > 0 and median_len < SHORT_NARRATIVE_PARAGRAPH_THRESHOLD:
            score -= min(
                20,
                (SHORT_NARRATIVE_PARAGRAPH_THRESHOLD - median_len)
                / SHORT_NARRATIVE_PARAGRAPH_THRESHOLD
                * 20,
            )

        return round(max(0, min(100, score)))


def check_sentence_variety(text: str) -> dict:
    """Check sentence length variety."""
    sentences = re.split(r"[。！？!?\n]", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) < 5:
        return {"uniform_sentences": False, "detail": "句子太少，无法分析"}

    lengths = [len(s) for s in sentences]
    consecutive_same = 0
    max_consecutive = 0
    for i in range(1, len(lengths)):
        diff_pct = abs(lengths[i] - lengths[i - 1]) / max(lengths[i - 1], 1)
        if diff_pct < 0.15:
            consecutive_same += 1
            max_consecutive = max(max_consecutive, consecutive_same)
        else:
            consecutive_same = 0

    return {
        "uniform_sentences": max_consecutive >= 3,
        "max_consecutive_same_length": max_consecutive,
        "detail": (
            f"最长连续{max_consecutive}个句子长度相近" if max_consecutive >= 3 else "句子长短有变化"
        ),
    }


def check_dialogue_ratio(text: str) -> dict:
    """Estimate dialogue ratio — descriptive metric, not a quality gate."""
    dialogue_parts = _DIALOGUE_RE.findall(text)
    dialogue_chars = sum(len(next((m for m in match if m), "")) for match in dialogue_parts)
    total_chars = len(text)
    ratio = dialogue_chars / total_chars if total_chars > 0 else 0
    return {
        "dialogue_ratio": round(ratio, 2),
        "detail": f"对话占比{ratio:.0%}",
    }


def check_ending(text: str) -> dict:
    """Check ending — summary patterns as evidence, hooks as neutral evidence."""
    last_paragraphs = [p.strip() for p in text.split("\n") if p.strip()][-3:]
    last_text = "\n".join(last_paragraphs)

    has_summary_ending = any(p in last_text for p in SUMMARY_PATTERNS)

    hook_evidence = [p for p in HOOK_EVIDENCE_PATTERNS if p in last_text]

    return {
        "summary_ending": has_summary_ending,
        "hook_evidence": hook_evidence,
        "detail": ("结尾总结式（应改为具体动作或悬念）" if has_summary_ending else "结尾中性"),
    }


# ── Style Gate ─────────────────────────────────────────


def style_gate(report: ParagraphStructureReport | dict) -> str:
    """Deterministic style gate — linter for paragraph structure.

    Returns PASS / WARNING / FAIL based on structural anomalies.
    Does NOT judge literary quality — that is the Editor's job.
    """
    if isinstance(report, dict):
        report = ParagraphStructureReport(**report)

    if report.max_consecutive_short_narrative_paragraphs > MAX_CONSECUTIVE_SHORT_PARAGRAPHS + 2:
        return "FAIL"
    if report.short_narrative_ratio > SHORT_NARRATIVE_RATIO_THRESHOLD + 0.2:
        return "FAIL"
    if report.single_sentence_narrative_ratio > SINGLE_SENTENCE_RATIO_THRESHOLD + 0.2:
        return "FAIL"
    if report.max_consecutive_short_narrative_paragraphs > MAX_CONSECUTIVE_SHORT_PARAGRAPHS:
        return "WARNING"
    if report.short_narrative_ratio > SHORT_NARRATIVE_RATIO_THRESHOLD:
        return "WARNING"
    if report.single_sentence_narrative_ratio > SINGLE_SENTENCE_RATIO_THRESHOLD:
        return "WARNING"
    return "PASS"


# ── Main detection ────────────────────────────────────


def detect_ai_flavor(text: str) -> dict:
    """Run all AI flavor detection rules and return a report.

    AI flavor is evidence/warning — banned phrases, clichés, and sentence
    patterns are recorded as issues. Paragraph structure is analyzed
    separately via ParagraphStructureAnalyzer. Dialogue ratio is descriptive.
    """
    issues = []

    for phrase in BANNED_CONNECTORS + BANNED_EMPHASIS:
        count = text.count(phrase)
        if count > 0:
            issues.append(
                {
                    "type": "banned_phrase",
                    "severity": "major",
                    "phrase": phrase,
                    "count": count,
                }
            )

    for phrase in BANNED_CLICHES:
        count = text.count(phrase)
        if count > 0:
            issues.append(
                {
                    "type": "cliche",
                    "severity": "minor",
                    "phrase": phrase,
                    "count": count,
                }
            )

    for pattern, label in BANNED_SENTENCE_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(
                {
                    "type": "sentence_pattern",
                    "severity": "minor",
                    "pattern": label,
                    "count": len(matches),
                }
            )

    para_report = ParagraphStructureAnalyzer().analyze(text)
    sent_check = check_sentence_variety(text)
    dialogue_check = check_dialogue_ratio(text)
    ending_check = check_ending(text)

    base_score = 100
    for issue in issues:
        deduction = 10 if issue["severity"] == "major" else 3
        base_score -= min(deduction * issue.get("count", 1), 30)
    if sent_check.get("uniform_sentences"):
        base_score -= 10
    if ending_check.get("summary_ending"):
        base_score -= 15

    score = max(0, base_score)

    return {
        "overall_score": score,
        "issues": issues,
        "paragraph_analysis": para_report.model_dump(),
        "sentence_analysis": sent_check,
        "dialogue_analysis": dialogue_check,
        "ending_analysis": ending_check,
        "total_issues": len(issues),
    }


class StyleAnalyzer:
    """Single entry point for deterministic style analysis.

    Combines:
    ├── AI Flavor Evidence (banned phrases, clichés, patterns)
    ├── Paragraph Structure (fragmentation, short narrative, consecutive)
    ├── Sentence Rhythm (length variety)
    └── Dialogue Statistics (descriptive)
    """

    def analyze(self, text: str, profile: StyleProfile | None = None) -> StyleReport:
        del profile
        legacy = detect_ai_flavor(text)
        para = legacy["paragraph_analysis"]
        sentence = legacy["sentence_analysis"]
        dialogue = legacy["dialogue_analysis"]
        return StyleReport(
            ai_flavor_score=legacy["overall_score"],
            paragraph_structure_score=para.get("fragmentation_score", 100),
            sentence_rhythm_score=60 if sentence.get("uniform_sentences") else 90,
            dialogue_score=dialogue.get("dialogue_ratio", 0) * 100,
            issues=legacy["issues"],
            paragraph_structure=para,
        )

    def legacy_report(self, text: str) -> dict:
        """Return the legacy detect_ai_flavor payload without changing its contract."""
        return detect_ai_flavor(text)
