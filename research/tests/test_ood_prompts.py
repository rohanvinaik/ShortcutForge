"""Tests for scripts/build_ood_prompts.py — JSONL parsing and prompt extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow importing research.src and scripts
# ---------------------------------------------------------------------------
_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RESEARCH_ROOT.parent
_SCRIPTS = _RESEARCH_ROOT / "scripts"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_ood_prompts import (
    _extract_user_message,
    _extract_user_prompts,
    _parse_jsonl_record,
)

# ---------------------------------------------------------------------------
# Tests: _parse_jsonl_record
# ---------------------------------------------------------------------------


class TestParseJsonlRecord:
    """Test safe JSON record parsing."""

    def test_returns_dict_for_valid_json(self):
        """Valid JSON line should return a dict."""
        line = '{"messages": [{"role": "user", "content": "hello"}]}'
        result = _parse_jsonl_record(line)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_returns_none_for_invalid_json(self):
        """Invalid JSON should return None, not raise."""
        result = _parse_jsonl_record("{not valid json}")
        assert result is None

    def test_returns_none_for_empty_line(self):
        """Empty string should return None."""
        assert _parse_jsonl_record("") is None

    def test_returns_none_for_whitespace_only(self):
        """Whitespace-only line should return None."""
        assert _parse_jsonl_record("   \t  \n") is None

    def test_strips_surrounding_whitespace(self):
        """Should handle JSON with surrounding whitespace."""
        line = '  {"key": "value"}  \n'
        result = _parse_jsonl_record(line)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Tests: _extract_user_message
# ---------------------------------------------------------------------------


class TestExtractUserMessage:
    """Test user message extraction from a parsed record."""

    def test_extracts_user_content(self):
        """Should return the first user message content."""
        record = {
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Set a timer for 5 minutes"},
                {"role": "assistant", "content": "SHORTCUT..."},
            ]
        }
        result = _extract_user_message(record)
        assert result == "Set a timer for 5 minutes"

    def test_returns_none_for_no_user_message(self):
        """Record with no user-role message should return None."""
        record = {
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "assistant", "content": "SHORTCUT..."},
            ]
        }
        assert _extract_user_message(record) is None

    def test_returns_none_for_empty_messages(self):
        """Record with empty messages list should return None."""
        record = {"messages": []}
        assert _extract_user_message(record) is None

    def test_returns_none_for_missing_messages_key(self):
        """Record without messages key should return None."""
        record = {"other_key": "value"}
        assert _extract_user_message(record) is None

    def test_returns_none_for_empty_user_content(self):
        """User message with empty content should return None."""
        record = {
            "messages": [
                {"role": "user", "content": ""},
            ]
        }
        assert _extract_user_message(record) is None

    def test_returns_none_for_whitespace_only_content(self):
        """User message with whitespace-only content should return None."""
        record = {
            "messages": [
                {"role": "user", "content": "   "},
            ]
        }
        assert _extract_user_message(record) is None


# ---------------------------------------------------------------------------
# Tests: _extract_user_prompts
# ---------------------------------------------------------------------------


class TestExtractUserPrompts:
    """Test end-to-end prompt extraction from a JSONL file."""

    def test_mixed_valid_and_invalid_records(self, tmp_path: Path):
        """Should extract prompts from valid records and skip invalid ones."""
        jsonl_file = tmp_path / "test_train.jsonl"
        lines = [
            # Valid record
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "Turn on Do Not Disturb"},
                        {"role": "assistant", "content": "DSL output"},
                    ]
                }
            ),
            # Invalid JSON
            "{broken json",
            # Empty line
            "",
            # Valid record with no user message
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "DSL output"},
                    ]
                }
            ),
            # Valid record
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Set a 5-minute timer"},
                    ]
                }
            ),
            # Whitespace-only line
            "   ",
        ]
        jsonl_file.write_text("\n".join(lines) + "\n")

        prompts = _extract_user_prompts(jsonl_file)
        assert len(prompts) == 2
        assert "Turn on Do Not Disturb" in prompts
        assert "Set a 5-minute timer" in prompts

    def test_empty_file(self, tmp_path: Path):
        """Empty JSONL file should return empty list."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        prompts = _extract_user_prompts(jsonl_file)
        assert prompts == []

    def test_all_invalid_records(self, tmp_path: Path):
        """File with only invalid records should return empty list."""
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text("{bad\n{also bad}\n")
        prompts = _extract_user_prompts(jsonl_file)
        assert prompts == []
