"""AI flavor detection — rule-based + LLM-assisted analysis.

Rules are adapted from Humanizer-zh (24 patterns) and community best practices,
scoped specifically for Chinese web novel writing.
"""

import re

from pydantic import BaseModel, ConfigDict, Field


class StyleProfile(BaseModel):
    """Optional style preferences reserved for future style-aware analysis."""


class StyleIssue(BaseModel):
    """A detected style issue with extensible evidence fields."""

    model_config = ConfigDict(extra="allow")

    type: str
    severity: str
    count: int = 1


class StyleReport(BaseModel):
    """Normalized style analysis result."""

    ai_flavor_score: float = Field(ge=0, le=100)
    paragraph_structure_score: float = Field(ge=0, le=100)
    sentence_rhythm_score: float = Field(ge=0, le=100)
    dialogue_score: float = Field(ge=0, le=100)
    issues: list[StyleIssue] = Field(default_factory=list)

# ── Banned phrases ────────────────────────────────────

BANNED_CONNECTORS = [
    "此外", "不仅如此", "更重要的是", "总而言之", "综上所述",
    "基于以上分析", "值得注意的是", "不难发现", "毫无疑问",
    "由此可见", "换言之", "也就是说",
]

BANNED_EMPHASIS = [
    "至关重要", "不可忽视", "深入探讨", "深刻揭示了",
    "具有重要的现实意义", "必须指出的是",
]

BANNED_CLICHES = [
    "他的眼中闪过一丝", "她的嘴角微微上扬", "他的心中涌起一股",
    "一种难以言喻的", "心中充满了", "眼中闪过一抹",
    "嘴角勾起一抹", "内心深处",
]

BANNED_SENTENCE_PATTERNS = [
    # "不是……而是……"翻案句 — OK in dialogue, flagged in narration
    (r"(?:这|那)?不是[^，。,\.]{1,30}而是[^，。,\.]{1,30}", "翻案句式「不是…而是…」"),
    # Overused 破折号 in narration
    (r"——", "破折号（叙事中建议削减）"),
    # 冒号讲义腔 (excluding dialogue markers like "说：")
    (r"(?:值得注意|必须承认|实事求是|坦白)地说", "冒号讲义腔"),
]


# ── Structural checks ─────────────────────────────────

def check_paragraph_lengths(text: str) -> dict:
    """Check for uniform paragraph lengths (AI tendency)."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) < 3:
        return {"uniform_paragraphs": False, "detail": "段落太少，无法分析"}

    lengths = [len(p) for p in paragraphs]
    avg_len = sum(lengths) / len(lengths)
    # If more than 60% of paragraphs are within 20% of average length
    variance_threshold = avg_len * 0.2
    uniform_count = sum(
        1 for length in lengths if abs(length - avg_len) < variance_threshold
    )
    ratio = uniform_count / len(lengths)

    return {
        "uniform_paragraphs": ratio > 0.6,
        "uniform_ratio": round(ratio, 2),
        "avg_paragraph_length": round(avg_len, 0),
        "detail": (
            f"{uniform_count}/{len(paragraphs)} 段落长度接近"
            if ratio > 0.6 else "段落长度有变化"
        ),
    }


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
        diff_pct = abs(lengths[i] - lengths[i-1]) / max(lengths[i-1], 1)
        if diff_pct < 0.15:  # Within 15% length = same
            consecutive_same += 1
            max_consecutive = max(max_consecutive, consecutive_same)
        else:
            consecutive_same = 0

    return {
        "uniform_sentences": max_consecutive >= 3,
        "max_consecutive_same_length": max_consecutive,
        "detail": (
            f"最长连续{max_consecutive}个句子长度相近"
            if max_consecutive >= 3 else "句子长短有变化"
        ),
    }


def check_dialogue_ratio(text: str) -> dict:
    """Estimate dialogue ratio by counting chars within paired quotes."""
    # Match content between paired Chinese/Western quotation marks
    dialogue_parts = re.findall(
        r'["]([^"]*?)["]|[「]([^」]*?)[」]|[『]([^』]*?)[』]',
        text,
    )
    dialogue_chars = sum(
        len(m[0] or m[1] or m[2]) for m in dialogue_parts
    )
    total_chars = len(text)
    ratio = dialogue_chars / total_chars if total_chars > 0 else 0
    return {
        "dialogue_ratio": round(ratio, 2),
        "ok": ratio >= 0.40,
        "detail": f"对话占比{ratio:.0%}" + ("（偏低，建议≥40%）" if ratio < 0.40 else "（合格）"),
    }


def check_ending(text: str) -> dict:
    """Check if chapter ending is a summary rather than a hook."""
    last_paragraphs = [p.strip() for p in text.split("\n") if p.strip()][-3:]
    last_text = "\n".join(last_paragraphs)

    summary_patterns = [
        "总之", "总而言之", "通过", "这次经历", "这天的经历",
        "他明白了", "她终于明白", "他学到了", "她意识到",
    ]
    has_summary_ending = any(p in last_text for p in summary_patterns)

    # Check if ending has a hook (question, cliffhanger, sudden event)
    hook_patterns = ["?", "？", "突然", "忽然", "就在这时", "紧接着", "就在此时", "下一秒"]

    has_hook = any(p in last_text for p in hook_patterns)

    return {
        "summary_ending": has_summary_ending,
        "has_hook": has_hook,
        "detail": (
            "结尾总结式（应改为具体动作或悬念）" if has_summary_ending
            else "结尾有钩子" if has_hook
            else "结尾中性，建议增加悬念"
        ),
    }


# ── Main detection ────────────────────────────────────

def detect_ai_flavor(text: str) -> dict:
    """Run all AI flavor detection rules and return a report."""
    issues = []

    # 1. Banned phrases
    for phrase in BANNED_CONNECTORS + BANNED_EMPHASIS:
        count = text.count(phrase)
        if count > 0:
            issues.append({
                "type": "banned_phrase",
                "severity": "major",
                "phrase": phrase,
                "count": count,
            })

    # 2. Clichés
    for phrase in BANNED_CLICHES:
        count = text.count(phrase)
        if count > 0:
            issues.append({
                "type": "cliche",
                "severity": "minor",
                "phrase": phrase,
                "count": count,
            })

    # 3. Sentence patterns
    for pattern, label in BANNED_SENTENCE_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            issues.append({
                "type": "sentence_pattern",
                "severity": "minor",
                "pattern": label,
                "count": len(matches),
            })

    # 4. Structural checks
    para_check = check_paragraph_lengths(text)
    sent_check = check_sentence_variety(text)
    dialogue_check = check_dialogue_ratio(text)
    ending_check = check_ending(text)

    # 5. Scoring
    base_score = 100
    for issue in issues:
        deduction = 10 if issue["severity"] == "major" else 3
        base_score -= min(deduction * issue.get("count", 1), 30)
    if para_check.get("uniform_paragraphs"):
        base_score -= 10
    if sent_check.get("uniform_sentences"):
        base_score -= 10
    if not dialogue_check.get("ok"):
        base_score -= 5
    if ending_check.get("summary_ending"):
        base_score -= 15

    score = max(0, base_score)

    return {
        "overall_score": score,
        "issues": issues,
        "paragraph_analysis": para_check,
        "sentence_analysis": sent_check,
        "dialogue_analysis": dialogue_check,
        "ending_analysis": ending_check,
        "total_issues": len(issues),
    }


class StyleAnalyzer:
    """Single entry point for deterministic style analysis."""

    def analyze(self, text: str, profile: StyleProfile | None = None) -> StyleReport:
        del profile  # The profile is an extension point; current rules are unchanged.
        legacy = detect_ai_flavor(text)
        paragraph = legacy["paragraph_analysis"]
        sentence = legacy["sentence_analysis"]
        dialogue = legacy["dialogue_analysis"]
        return StyleReport(
            ai_flavor_score=legacy["overall_score"],
            paragraph_structure_score=60 if paragraph.get("uniform_paragraphs") else 90,
            sentence_rhythm_score=60 if sentence.get("uniform_sentences") else 90,
            dialogue_score=dialogue.get("dialogue_ratio", 0) * 100,
            issues=legacy["issues"],
        )

    def legacy_report(self, text: str) -> dict:
        """Return the existing tool payload without changing its contract."""
        return detect_ai_flavor(text)
