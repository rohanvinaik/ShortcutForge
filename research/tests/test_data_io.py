#!/usr/bin/env python3
"""Tests for research.src.data — JSONL loading and malformed-row behavior."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.src.contracts import TypedIRExample
from research.src.data import load_typed_ir_jsonl, save_typed_ir_jsonl


class TestJSONLRoundtrip(unittest.TestCase):
    def _make_example(self, idx: int = 0):
        return TypedIRExample(
            shortcut_id=f"test-{idx:03d}",
            system_prompt="sys",
            prompt=f"prompt {idx}",
            dsl=f"DSL {idx}",
            shortcut_name=f"Name{idx}",
            tier1_tokens=["SHORTCUT", "ENDSHORTCUT"],
            tier2_blocks=[],
            tier3_slots=[],
        )

    def test_roundtrip(self):
        examples = [self._make_example(i) for i in range(5)]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        save_typed_ir_jsonl(examples, path)
        loaded = load_typed_ir_jsonl(path)
        self.assertEqual(len(loaded), 5)
        self.assertEqual(loaded[0].shortcut_id, "test-000")
        self.assertEqual(loaded[4].prompt, "prompt 4")
        path.unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("")
            path = Path(f.name)
        loaded = load_typed_ir_jsonl(path)
        self.assertEqual(len(loaded), 0)
        path.unlink()

    def test_blank_lines_skipped(self):
        ex = self._make_example(0)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("\n")
            f.write(json.dumps(ex.to_dict()) + "\n")
            f.write("\n")
            f.write(json.dumps(ex.to_dict()) + "\n")
            path = Path(f.name)
        loaded = load_typed_ir_jsonl(path)
        self.assertEqual(len(loaded), 2)
        path.unlink()

    def test_malformed_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("not valid json\n")
            path = Path(f.name)
        with self.assertRaises(ValueError):
            load_typed_ir_jsonl(path)
        path.unlink()

    def test_missing_field_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write('{"shortcut_id": "test"}\n')  # Missing required fields
            path = Path(f.name)
        with self.assertRaises(ValueError):
            load_typed_ir_jsonl(path)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
