"""Tests for ContextCompressor."""

from novel_agent.memory.compressor import (
    CompressionStrategy,
    _build_simple_summary,
    estimate_tokens,
    extract_critical_snippets,
)


class TestEstimateTokens:
    def test_chinese_text(self):
        # Chinese: ~1.5 chars/token
        tokens = estimate_tokens("这是一段中文测试文本" * 10)
        assert tokens > 0
        # 10 chars * 10 = 100 chars / 1.5 ≈ 66 tokens
        assert 50 < tokens < 80

    def test_english_text(self):
        # English: ~4 chars/token
        tokens = estimate_tokens("hello world test text " * 10)
        assert tokens > 0

    def test_mixed_text(self):
        tokens = estimate_tokens("你好hello世界world测试test")
        assert tokens > 0

    def test_empty_text(self):
        assert estimate_tokens("") == 0


class TestExtractCriticalSnippets:
    def test_extracts_foreshadowing(self):
        text = "这是一段普通内容。这里有个伏笔需要注意。另外的内容。"
        patterns = [r"伏笔"]
        snippets = extract_critical_snippets(text, patterns)
        assert any("伏笔" in s for s in snippets)

    def test_limits_to_10(self):
        text = "这里有规则。还有设定。还有体系。" * 20
        patterns = [r"规则|设定|体系"]
        snippets = extract_critical_snippets(text, patterns)
        assert len(snippets) <= 10

    def test_no_matches(self):
        text = "普通内容没有任何关键信息。"
        patterns = [r"伏笔"]
        snippets = extract_critical_snippets(text, patterns)
        assert snippets == []


class TestCompressionStrategy:
    def test_defaults(self):
        strategy = CompressionStrategy()
        assert strategy.trigger_threshold == 40000
        assert strategy.target_tokens == 20000
        assert len(strategy.preserve_patterns) == 4

    def test_custom_thresholds(self):
        strategy = CompressionStrategy(
            trigger_threshold=10000,
            target_tokens=5000,
        )
        assert strategy.trigger_threshold == 10000
        assert strategy.target_tokens == 5000


class TestBuildSimpleSummary:
    def test_builds_from_chapters(self):
        chapters = [
            {"chapter_number": 1, "draft_content": "第一章正文内容" * 5},
            {"chapter_number": 2, "draft_content": "第二章正文内容" * 5},
        ]
        summary = _build_simple_summary(chapters)
        assert "第1章" in summary
        assert "第2章" in summary

    def test_empty_chapters(self):
        assert _build_simple_summary([]) == ""

    def test_truncates_long_content(self):
        chapters = [
            {"chapter_number": 1, "draft_content": "X" * 500},
        ]
        summary = _build_simple_summary(chapters)
        assert "..." in summary
        assert len(summary) < 500
