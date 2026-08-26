"""Tests for style analysis — paragraph structure, AI flavor evidence, style gate."""

from pathlib import Path

from novel_agent.style.analyzer import (
    ParagraphStructureAnalyzer,
    StyleAnalyzer,
    check_dialogue_ratio,
    check_ending,
    check_sentence_variety,
    detect_ai_flavor,
    style_gate,
)

FIXTURES = Path(__file__).parent / "fixtures" / "style"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestBannedPhrases:
    def test_detects_banned_connectors(self):
        text = "此外，这是一个问题。不仅如此，还有更多。"
        report = detect_ai_flavor(text)
        phrases = [i["phrase"] for i in report["issues"]]
        assert "此外" in phrases
        assert "不仅如此" in phrases

    def test_detects_banned_emphasis(self):
        text = "这个发现至关重要，不可忽视。"
        report = detect_ai_flavor(text)
        phrases = [i["phrase"] for i in report["issues"]]
        assert "至关重要" in phrases
        assert "不可忽视" in phrases

    def test_detects_cliches(self):
        text = "他的眼中闪过一丝惊讶，她的嘴角微微上扬。"
        report = detect_ai_flavor(text)
        cliches = [i["phrase"] for i in report["issues"] if i["type"] == "cliche"]
        assert "他的眼中闪过一丝" in cliches

    def test_clean_text_scores_high(self):
        text = (
            '"你来了。"林风推开包厢的门，目光扫过在场的人。\n'
            "王胖子一拍桌子站起来，脸上横肉抖动。\n"
            '"东西呢？"\n'
            "林风从口袋里摸出一个U盘，扔在桌上。\n"
            '"自己看。"\n'
            "王胖子接过去，插进电脑。屏幕亮起来的瞬间，他的脸色变了。\n"
        )
        report = detect_ai_flavor(text)
        assert report["overall_score"] >= 80

    def test_multiple_issues_reduce_score(self):
        text = (
            "此外，这个方案至关重要。不仅如此，更重要的是，"
            "我们必须深入探讨这个问题的本质。他的眼中闪过一丝疑惑，"
            "她的嘴角微微上扬。综上所述，这一切都具有重要的现实意义。"
        )
        report = detect_ai_flavor(text)
        assert report["overall_score"] < 80
        assert report["total_issues"] > 0

    def test_style_analyzer_returns_normalized_report(self):
        report = StyleAnalyzer().analyze('"你好。"林风转身。')

        assert 0 <= report.ai_flavor_score <= 100
        assert 0 <= report.paragraph_structure_score <= 100
        assert 0 <= report.sentence_rhythm_score <= 100
        assert report.dialogue_score > 0
        assert isinstance(report.issues, list)
        assert report.paragraph_structure is not None
        assert report.sentence_rhythm is not None
        assert report.dialogue_stats is not None
        assert report.ending_analysis is not None
        assert report.style_gate in ("PASS", "WARNING", "FAIL")

    def test_legacy_detect_ai_flavor_matches_analyzer(self):
        """detect_ai_flavor() is a thin wrapper around StyleAnalyzer.analyze()."""
        text = "此外，这个发现至关重要。"
        report = StyleAnalyzer().analyze(text)
        legacy = detect_ai_flavor(text)

        assert legacy["overall_score"] == report.ai_flavor_score
        assert len(legacy["issues"]) == len(report.issues)
        assert legacy["paragraph_analysis"] == report.paragraph_structure
        assert legacy["sentence_analysis"] == report.sentence_rhythm
        assert legacy["dialogue_analysis"] == report.dialogue_stats
        assert legacy["ending_analysis"] == report.ending_analysis


class TestParagraphStructure:
    def test_fragmented_text_has_low_score(self):
        text = _fixture("fragmented.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.fragmentation_score < 60
        assert report.max_consecutive_short_narrative_paragraphs >= 4
        assert report.short_narrative_ratio > 0.4

    def test_natural_text_has_high_score(self):
        text = _fixture("natural.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.fragmentation_score >= 70
        assert report.max_consecutive_short_narrative_paragraphs <= 2

    def test_dialogue_heavy_not_flagged_as_fragmented(self):
        text = _fixture("dialogue_heavy.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        # Fragmentation score should remain healthy — short narrative beats
        # between dialogue are natural, not fragmentation.
        assert report.fragmentation_score >= 60
        assert report.dialogue_paragraph_count >= 5
        # Pure dialogue paragraphs must be excluded from narrative metrics.
        assert report.max_consecutive_short_narrative_paragraphs <= 2

    def test_action_scene_short_paragraphs_not_over_penalized(self):
        text = _fixture("action_scene.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.fragmentation_score >= 30
        assert report.narrative_paragraph_count > 0

    def test_mixed_scene_balanced(self):
        text = _fixture("mixed_scene.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.fragmentation_score >= 50
        assert report.mixed_paragraph_count > 0

    def test_empty_text(self):
        report = ParagraphStructureAnalyzer().analyze("")
        assert report.paragraph_count == 0
        assert report.fragmentation_score == 100

    def test_single_sentence_detection(self):
        text = "他转过身。\n这是一个比较长的叙述段落，包含了更多内容来避免被判定为短段。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.single_sentence_narrative_ratio > 0

    def test_pure_dialogue_not_counted_as_short_narrative(self):
        text = (
            '"走。"\n"好。"\n"来。"\n"行。"\n'
            "这是最后一个足够长的叙述段落，用来确保前面那些纯对白不被计入短叙述段比例，"
            "这一段的长度超过了四十个字符的阈值。"
        )
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.dialogue_paragraph_count == 4
        assert report.narrative_paragraph_count == 1
        assert report.short_narrative_ratio == 0

    def test_consecutive_short_narrative(self):
        text = "他抬头。\n雨停了。\n街道很安静。\n远处有人走来。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.max_consecutive_short_narrative_paragraphs == 4

    def test_scene_boundary_not_phantom_paragraph(self):
        text = (
            "第一段场景内容很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长。\n\n"
            "第二段场景内容也很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长。"
        )
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.paragraph_count == 2
        assert report.short_narrative_ratio == 0

    def test_chinese_curly_quotes(self):
        text = "\u201c你好。\u201d林风说道。这里有一些叙述内容。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.mixed_paragraph_count == 1

    def test_corner_brackets(self):
        text = "「你好。」林风说道。这里有一些叙述内容。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.mixed_paragraph_count == 1

    def test_white_corner_brackets(self):
        text = "『你好。』林风说道。这里有一些叙述内容。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.mixed_paragraph_count == 1

    def test_nested_quotes(self):
        text = "\u201c他说：\u2018走。\u2019\u201d"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.dialogue_paragraph_count == 1


class TestSentenceVariety:
    def test_consecutive_same_length_detected(self):
        text = (
            "这是一个十五字句子啊啊啊啊。"
            "这也是十五字句子呢呢呢。"
            "这也是个十五字句子了。"
            "还是十五字的句子哟哟。"
            "又是十五字句子耶耶耶。"
        )
        result = check_sentence_variety(text)
        assert result["uniform_sentences"] is True
        assert result["max_consecutive_same_length"] >= 3

    def test_varied_sentences_pass(self):
        text = "短句。这是一个中等长度的句子来增加变化。"
        "很长很长的句子" + "内容" * 50 + "。短。"
        "又一句。然后这里有一段比较长的内容" + "文字" * 20 + "。"
        result = check_sentence_variety(text)
        assert result["uniform_sentences"] is False

    def test_few_sentences_not_analyzed(self):
        text = "一句。两句。"
        result = check_sentence_variety(text)
        assert "句子太少" in result["detail"]


class TestDialogueRatio:
    def test_high_dialogue_detected(self):
        dialogue_heavy = (
            '"你来了。" "来了。" "东西呢？" "在这。" '
            '"不可能。" "你自己看。" "真的是它。" "我早就知道。" '
            '"接下来怎么办？" "等。" "等什么？" "等他出现。"'
        )
        result = check_dialogue_ratio(dialogue_heavy)
        assert result["dialogue_ratio"] >= 0.40
        assert "ok" not in result

    def test_low_dialogue_detected(self):
        text = "纯叙述内容没有任何对话标记。" * 20
        result = check_dialogue_ratio(text)
        assert result["dialogue_ratio"] < 0.40
        assert "ok" not in result

    def test_chinese_quotes_detected(self):
        text = "「你好。」林风说道。这里有一些叙述内容。" * 10
        result = check_dialogue_ratio(text)
        assert result["dialogue_ratio"] > 0

    def test_white_corner_quotes_detected(self):
        text = "『你来了。』林风说道。这里有一些叙述内容。" * 10
        result = check_dialogue_ratio(text)
        assert result["dialogue_ratio"] > 0

    def test_empty_text(self):
        result = check_dialogue_ratio("")
        assert result["dialogue_ratio"] == 0


class TestEnding:
    def test_summary_ending_detected(self):
        text = "前面内容\n" * 10 + "总之，通过这次经历，他学到了很多。"
        result = check_ending(text)
        assert result["summary_ending"] is True

    def test_hook_evidence_recorded_not_rewarded(self):
        text = "前面内容\n" * 10 + "就在这时，门外突然传来一声巨响。"
        result = check_ending(text)
        assert "hook_evidence" in result
        assert len(result["hook_evidence"]) > 0
        assert "has_hook" not in result

    def test_neutral_ending(self):
        text = "前面内容\n" * 10 + "他转身离开了房间。"
        result = check_ending(text)
        assert not result["summary_ending"]
        assert result["hook_evidence"] == []


class TestStyleGate:
    def _report(self, **kwargs):
        from novel_agent.style.analyzer import ParagraphStructureReport

        defaults = dict(
            paragraph_count=10,
            narrative_paragraph_count=8,
            dialogue_paragraph_count=2,
            mixed_paragraph_count=0,
            median_narrative_paragraph_length=100,
            short_narrative_ratio=0.1,
            single_sentence_narrative_ratio=0.1,
            max_consecutive_short_narrative_paragraphs=1,
            fragmentation_score=90,
            issues=[],
        )
        defaults.update(kwargs)
        return ParagraphStructureReport(**defaults)

    def test_pass_on_natural(self):
        assert style_gate(self._report()) == "PASS"

    def test_warning_on_moderate_short_ratio(self):
        assert style_gate(self._report(short_narrative_ratio=0.45)) == "WARNING"

    def test_fail_on_extreme_short_ratio(self):
        assert style_gate(self._report(short_narrative_ratio=0.65)) == "FAIL"

    def test_warning_on_consecutive_shorts(self):
        assert style_gate(self._report(max_consecutive_short_narrative_paragraphs=4)) == "WARNING"

    def test_fail_on_extreme_consecutive_shorts(self):
        assert style_gate(self._report(max_consecutive_short_narrative_paragraphs=6)) == "FAIL"

    def test_warning_on_single_sentence_ratio(self):
        assert style_gate(self._report(single_sentence_narrative_ratio=0.35)) == "WARNING"

    def test_fail_on_extreme_single_sentence_ratio(self):
        assert style_gate(self._report(single_sentence_narrative_ratio=0.55)) == "FAIL"

    def test_fragmented_fixture_fails(self):
        text = _fixture("fragmented.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert style_gate(report) in ("WARNING", "FAIL")

    def test_natural_fixture_passes(self):
        text = _fixture("natural.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert style_gate(report) == "PASS"
