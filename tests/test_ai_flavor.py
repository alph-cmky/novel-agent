"""Tests for AI flavor detection rules."""

from novel_agent.style.ai_flavor import (
    check_dialogue_ratio,
    check_ending,
    check_paragraph_lengths,
    check_sentence_variety,
    detect_ai_flavor,
)


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
        cliches = [
            i["phrase"] for i in report["issues"] if i["type"] == "cliche"
        ]
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


class TestParagraphLengths:
    def test_uniform_paragraphs_detected(self):
        # Four paragraphs with nearly identical lengths
        text = "\n".join(["A" * 100 for _ in range(10)])
        result = check_paragraph_lengths(text)
        assert result["uniform_paragraphs"] is True

    def test_varied_paragraphs_pass(self):
        text = "\n".join([
            "A" * 20,
            "B" * 200,
            "C" * 50,
            "D" * 120,
            "E" * 80,
        ])
        result = check_paragraph_lengths(text)
        assert result["uniform_paragraphs"] is False

    def test_few_paragraphs_not_analyzed(self):
        text = "A\nB"
        result = check_paragraph_lengths(text)
        assert "段落太少" in result["detail"]


class TestSentenceVariety:
    def test_consecutive_same_length_detected(self):
        # 5 sentences with very similar lengths
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
    def test_high_dialogue_passes(self):
        # Heavy dialogue, minimal narration
        dialogue_heavy = (
            '"你来了。" "来了。" "东西呢？" "在这。" '
            '"不可能。" "你自己看。" "真的是它。" "我早就知道。" '
            '"接下来怎么办？" "等。" "等什么？" "等他出现。"'
        )
        result = check_dialogue_ratio(dialogue_heavy)
        assert result["ok"] is True
        assert result["dialogue_ratio"] >= 0.40

    def test_low_dialogue_flags(self):
        text = "纯叙述内容没有任何对话标记。" * 20
        result = check_dialogue_ratio(text)
        assert result["ok"] is False
        assert result["dialogue_ratio"] < 0.40

    def test_chinese_quotes_detected(self):
        text = "「你好。」林风说道。这里有一些叙述内容。" * 10
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

    def test_hook_ending_detected(self):
        text = "前面内容\n" * 10 + "就在这时，门外突然传来一声巨响。"
        result = check_ending(text)
        assert result["has_hook"] is True

    def test_neutral_ending(self):
        text = "前面内容\n" * 10 + "他转身离开了房间。"
        result = check_ending(text)
        assert not result["summary_ending"]
