"""Tests for the shared JSON parser — the None-sanitizing boundary."""

from novel_agent.schema.parser import parse_json_response, strip_none


class TestParseJsonResponse:
    def test_direct_parse_strips_nested_none(self):
        text = (
            '{"chapter_strategy": {"tension_profile": '
            '{"variety_check": null, "chapter_tension": 7}}}'
        )
        result = parse_json_response(text)
        assert result["chapter_strategy"]["tension_profile"] == {"chapter_tension": 7}

    def test_markdown_block_strips_none(self):
        text = '```json\n{"a": 1, "b": null}\n```'
        result = parse_json_response(text)
        assert result == {"a": 1}

    def test_defaults_are_sanitized_too(self):
        """Placeholder None fields in defaults must be dropped, not kept."""
        result = parse_json_response(
            "not json at all",
            defaults={"keep": 1, "chapter_strategy": {"tension_profile": None, "pacing": "normal"}},
        )
        assert result["keep"] == 1
        assert result["chapter_strategy"] == {"pacing": "normal"}
        assert result["raw_output"] == "not json at all"

    def test_non_dict_parse_falls_back_to_defaults(self):
        """json.loads('null') succeeds but isn't a dict — must fall back."""
        result = parse_json_response("null", defaults={"ok": True})
        assert result["ok"] is True

    def test_empty_text_uses_sanitized_defaults(self):
        result = parse_json_response("", defaults={"x": None, "y": 2})
        assert result == {"y": 2}


class TestStripNone:
    def test_scalars_pass_through(self):
        assert strip_none("s") == "s"
        assert strip_none(0) == 0
        assert strip_none(False) is False


class TestRepair:
    def test_missing_opening_quote_on_value(self):
        text = '{"entity_type": "item", "properties": {"数量": 10枚", "用途": "未说明"}}'
        result = parse_json_response(text)
        assert result["properties"]["数量"] == "10枚"
        assert result["properties"]["用途"] == "未说明"

    def test_trailing_comma(self):
        text = '{"a": 1, "b": [1, 2,],}'
        result = parse_json_response(text)
        assert result == {"a": 1, "b": [1, 2]}

    def test_repair_inside_markdown_block(self):
        text = (
            '```json\n{"new_entities": [{"name": "模拟币", "数量": 10枚"}],'
            ' "conflicts": []}\n```'
        )
        result = parse_json_response(text)
        assert result["new_entities"][0]["数量"] == "10枚"

    def test_repair_does_not_corrupt_valid_values(self):
        text = '{"age": 25, "score": 85.5, "flag": true, "name": "林砚", "nested": {"x": null}}'
        result = parse_json_response(text)
        assert result["age"] == 25
        assert result["score"] == 85.5
        assert result["flag"] is True
        assert result["name"] == "林砚"
        assert "x" not in result["nested"]

    def test_unrepairable_falls_back_to_defaults(self):
        text = '{"a": totally broken {'
        result = parse_json_response(text, defaults={"fallback": 1})
        assert result["fallback"] == 1
        assert result["raw_output"] == text
