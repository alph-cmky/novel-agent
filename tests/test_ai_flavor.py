"""Tests for style analysis — paragraph structure, AI flavor evidence, style gate."""

from pathlib import Path

from novel_agent.style.analyzer import (
    ParagraphStructureAnalyzer,
    StyleAnalyzer,
    check_dialogue_ratio,
    check_ending,
    check_sentence_variety,
    style_gate,
)

FIXTURES = Path(__file__).parent / "fixtures" / "style"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestBannedPhrases:
    def test_detects_banned_connectors(self):
        text = "此外，这是一个问题。不仅如此，还有更多。"
        report = StyleAnalyzer().analyze(text)
        phrases = [i.phrase for i in report.issues if hasattr(i, "phrase")]
        assert "此外" in phrases
        assert "不仅如此" in phrases

    def test_detects_banned_emphasis(self):
        text = "这个发现至关重要，不可忽视。"
        report = StyleAnalyzer().analyze(text)
        phrases = [i.phrase for i in report.issues if hasattr(i, "phrase")]
        assert "至关重要" in phrases
        assert "不可忽视" in phrases

    def test_detects_cliches(self):
        text = "他的眼中闪过一丝惊讶，她的嘴角微微上扬。"
        report = StyleAnalyzer().analyze(text)
        cliches = [i.phrase for i in report.issues if hasattr(i, "phrase") and i.type == "cliche"]
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
        report = StyleAnalyzer().analyze(text)
        assert report.ai_flavor_score >= 80

    def test_multiple_issues_reduce_score(self):
        text = (
            "此外，这个方案至关重要。不仅如此，更重要的是，"
            "我们必须深入探讨这个问题的本质。他的眼中闪过一丝疑惑，"
            "她的嘴角微微上扬。综上所述，这一切都具有重要的现实意义。"
        )
        report = StyleAnalyzer().analyze(text)
        assert report.ai_flavor_score < 80
        assert len(report.issues) > 0

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

    def test_dialogue_driven_extreme_short_ratio_not_fail(self):
        """Phase K: 对白驱动场景（≥40% 纯对白段）豁免 short-ratio 硬线。

        对白行之间的短叙述拍是自然节奏，short_ratio 83% 不构成 FAIL。
        """
        report = self._report(
            paragraph_count=18,
            narrative_paragraph_count=3,
            dialogue_paragraph_count=12,
            mixed_paragraph_count=3,
            short_narrative_ratio=0.83,
            max_consecutive_short_narrative_paragraphs=2,
        )
        assert style_gate(report) == "PASS"

    def test_dialogue_driven_consecutive_shorts_still_fail(self):
        """对话场景中连续短段检查仍然生效——对白豁免不等于放弃结构检查。"""
        report = self._report(
            paragraph_count=18,
            narrative_paragraph_count=3,
            dialogue_paragraph_count=12,
            mixed_paragraph_count=3,
            max_consecutive_short_narrative_paragraphs=6,
        )
        assert style_gate(report) == "FAIL"

    def test_low_dialogue_share_not_exempt(self):
        """对白段占比 < 40% 不触发豁免——极端 short_ratio 仍 FAIL。"""
        report = self._report(short_narrative_ratio=0.65)
        assert style_gate(report) == "FAIL"


class TestFixtureMetricAlignment:
    """Phase K2: 6 类中文 fixture 的人工评价 ↔ Analyzer 指标对齐。

    人工评价编码为断言——目标是 Analyzer 指标能解释人工评价：
    - fragmented:        段落自然度差（单句段 + 连续短段），但无 AI 味用词
    - natural:           各维度健康，叙事对白交织
    - dialogue_heavy:    对白主导，短叙述拍是对白节奏的自然组成
    - action:            动作短段合理，不应被过度惩罚
    - description_heavy: 描写流，无对白，句子节奏偏平
    - mixed:             叙事 + 对白平衡
    """

    @staticmethod
    def _analyze(fixture: str) -> "StyleAnalyzer":
        from novel_agent.style.analyzer import StyleAnalyzer

        return StyleAnalyzer().analyze(_fixture(fixture))

    def test_fragmented(self):
        """人工：段落自然度极差 → frag 低 + gate FAIL + 连续短段证据。"""
        r = self._analyze("fragmented.txt")
        assert r.paragraph_structure_score < 40
        assert r.style_gate == "FAIL"
        ps = r.paragraph_structure
        assert ps["short_narrative_ratio"] >= 0.9
        assert ps["max_consecutive_short_narrative_paragraphs"] >= 5

    def test_fragmented_is_structural_not_ai_flavor(self):
        """人工：碎片化是结构问题，不是 AI 味 → ai_flavor 不受段落结构影响。"""
        r = self._analyze("fragmented.txt")
        assert r.ai_flavor_score == 100
        assert r.issues == []

    def test_natural(self):
        """人工：各维度健康 → gate PASS + frag 高 + 对白占比适中。"""
        r = self._analyze("natural.txt")
        assert r.style_gate == "PASS"
        assert r.paragraph_structure_score >= 90
        assert r.ai_flavor_score >= 90
        assert 0.1 <= r.dialogue_score / 100 <= 0.3

    def test_dialogue_heavy(self):
        """人工：对白主导场景，短拍自然 → dlg 高 + gate 不误报 FAIL。"""
        r = self._analyze("dialogue_heavy.txt")
        assert r.dialogue_score >= 30
        assert r.style_gate == "PASS"
        ps = r.paragraph_structure
        assert ps["dialogue_paragraph_count"] >= 8
        assert ps["max_consecutive_short_narrative_paragraphs"] <= 3

    def test_action(self):
        """人工：动作短段是节奏需要 → frag 不被过度惩罚。"""
        r = self._analyze("action_scene.txt")
        assert r.paragraph_structure_score >= 60
        assert r.paragraph_structure["narrative_paragraph_count"] > 0
        assert r.dialogue_score < 10

    def test_description_heavy(self):
        """人工：描写流、零对白、句子节奏偏平 → 指标三项全部解释。"""
        r = self._analyze("description_heavy.txt")
        assert r.dialogue_score < 5
        assert r.paragraph_structure_score >= 90
        assert r.style_gate == "PASS"
        assert r.paragraph_structure["median_narrative_paragraph_length"] >= 80
        # 人工「节奏偏平」由 sentence_rhythm 的 uniform 检测解释
        assert r.sentence_rhythm["uniform_sentences"] is True

    def test_mixed(self):
        """人工：叙事 + 对白平衡 → mixed 段存在 + 结构健康。"""
        r = self._analyze("mixed_scene.txt")
        assert r.paragraph_structure["mixed_paragraph_count"] >= 3
        assert r.paragraph_structure_score >= 70
        assert 0 < r.dialogue_score < 15

    def test_fragmentation_ranking_matches_human(self):
        """人工段落自然度排序: natural/description/mixed/action > dialogue > fragmented。"""
        frag = {
            name: self._analyze(f).paragraph_structure_score
            for name, f in [
                ("natural", "natural.txt"),
                ("description", "description_heavy.txt"),
                ("mixed", "mixed_scene.txt"),
                ("action", "action_scene.txt"),
                ("dialogue", "dialogue_heavy.txt"),
                ("fragmented", "fragmented.txt"),
            ]
        }
        assert (
            min(frag["natural"], frag["description"], frag["mixed"], frag["action"])
            > frag["dialogue"]
        )
        assert frag["dialogue"] > frag["fragmented"]

    def test_dialogue_ratio_ranking_matches_human(self):
        """人工对白占比排序: dialogue_heavy > natural > description_heavy。"""
        dlg = {
            name: self._analyze(f).dialogue_score
            for name, f in [
                ("dialogue", "dialogue_heavy.txt"),
                ("natural", "natural.txt"),
                ("description", "description_heavy.txt"),
            ]
        }
        assert dlg["dialogue"] > dlg["natural"] > dlg["description"]


class TestParagraphCalibration:
    """P2-1: 段落分类与短段判定的校准 — 综合判断，不做一刀切。"""

    def test_task_doc_alternation_sample(self):
        """任务文档 §23 样本：对白/叙述交替分类正确。"""
        from novel_agent.style.analyzer import _classify_paragraph

        assert _classify_paragraph("\u201c走。\u201d") == "DIALOGUE"
        assert _classify_paragraph("他转过身。") == "NARRATIVE"
        assert _classify_paragraph("\u201c等等。\u201d") == "DIALOGUE"
        assert _classify_paragraph("他没有回头。") == "NARRATIVE"

    def test_speech_tag_is_mixed(self):
        """说话人提示 + 引语 → MIXED（两种引号）。"""
        from novel_agent.style.analyzer import _classify_paragraph

        assert _classify_paragraph("他说：\u201c走。\u201d") == "MIXED"
        assert _classify_paragraph("她说：「等等。」") == "MIXED"

    def test_action_burst_inside_varied_prose_not_over_penalized(self):
        """任务文档 §24：整体节奏健康的章节中，孤立短句爆发不应重罚。

        连续短段惩罚必须与整体短段比例耦合 — 碎片化是多信号共振，
        单一局部爆发只是节奏选择。
        """
        long_para = (
            "他沿着河堤慢慢走了很久。一路上把心里那些翻来覆去的念头一条一条摆开来检视，"
            "又逐一把它们收回原处。最后只留下一句还没有答案的话悬在那里。"
        )
        burst = "他停步。\n风很紧。\n有人跟了上来。\n他攥紧了刀柄。"
        text = "\n\n".join([long_para] * 6) + "\n\n" + burst
        report = ParagraphStructureAnalyzer().analyze(text)

        assert report.max_consecutive_short_narrative_paragraphs >= 4
        # coupled penalty keeps the score healthy despite the local burst
        assert report.fragmentation_score >= 80

    def test_kinetic_beats_reported_as_evidence_only(self):
        """任务文档 §24 四连拍全部计入 kinetic_beat_count，仅作证据。"""
        text = "他拔刀。\n刀光一闪。\n他退后一步。\n对方已经逼近。"
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.kinetic_beat_count == 4
        assert report.narrative_paragraph_count == 4

    def test_chapter_wide_fragmentation_still_severe(self):
        """全章碎片化（多信号共振）仍被判严重 — 耦合不放松真问题。"""
        text = _fixture("fragmented.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.fragmentation_score < 60
        assert style_gate(report) in ("WARNING", "FAIL")

    def test_beat_count_zero_on_natural_prose(self):
        """自然长段无 beat 噪声计数。"""
        text = _fixture("natural.txt")
        report = ParagraphStructureAnalyzer().analyze(text)
        assert report.kinetic_beat_count == 0
