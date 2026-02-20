"""
Tests for label_training_data.py — Training Data Labeler.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_TRAINING_DIR = _SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(_TRAINING_DIR))

from label_training_data import TrainingDataLabeler, _count_actions, _detect_constructs


class TestConstructDetection(unittest.TestCase):
    """Tests for DSL construct detection."""

    def test_detect_if(self):
        dsl = 'SHORTCUT "Test"\nACTION comment\nIF $x "is" "y"\nACTION alert\nENDIF'
        self.assertIn("IF", _detect_constructs(dsl))

    def test_detect_menu(self):
        dsl = 'SHORTCUT "Test"\nMENU "Pick"\nCASE "A"\nACTION alert\nENDMENU'
        self.assertIn("MENU", _detect_constructs(dsl))

    def test_detect_repeat(self):
        dsl = 'SHORTCUT "Test"\nREPEAT 5\nACTION alert\nENDREPEAT'
        self.assertIn("REPEAT", _detect_constructs(dsl))

    def test_detect_foreach(self):
        dsl = 'SHORTCUT "Test"\nFOREACH $items\nACTION alert\nENDFOREACH'
        self.assertIn("FOREACH", _detect_constructs(dsl))

    def test_detect_multiple(self):
        dsl = 'IF $x "is" "y"\nREPEAT 3\nFOREACH $z\nENDFOREACH\nENDREPEAT\nENDIF'
        constructs = _detect_constructs(dsl)
        self.assertIn("IF", constructs)
        self.assertIn("REPEAT", constructs)
        self.assertIn("FOREACH", constructs)
        self.assertNotIn("MENU", constructs)

    def test_detect_none(self):
        dsl = 'SHORTCUT "Test"\nACTION alert\nACTION showresult'
        self.assertEqual([], _detect_constructs(dsl))

    def test_no_false_positive_inline_if(self):
        """IF must be at line start, not inside action parameters."""
        dsl = 'ACTION comment WFCommentActionText="Check IF condition"'
        self.assertNotIn("IF", _detect_constructs(dsl))


class TestActionCounting(unittest.TestCase):
    """Tests for ACTION statement counting."""

    def test_count_single(self):
        dsl = 'SHORTCUT "Test"\nACTION alert'
        self.assertEqual(1, _count_actions(dsl))

    def test_count_multiple(self):
        dsl = 'SHORTCUT "Test"\nACTION alert\nACTION showresult\nACTION setclipboard'
        self.assertEqual(3, _count_actions(dsl))

    def test_count_zero(self):
        dsl = 'SHORTCUT "Test"\nSET $x = "hello"'
        self.assertEqual(0, _count_actions(dsl))

    def test_count_nested(self):
        dsl = 'SHORTCUT "Test"\nIF $x "is" "y"\nACTION alert\nELSE\nACTION showresult\nENDIF'
        self.assertEqual(2, _count_actions(dsl))


class TestTrainingDataLabeler(unittest.TestCase):
    """Tests for the TrainingDataLabeler class."""

    def setUp(self):
        self.labeler = TrainingDataLabeler()

    def _make_example(self, prompt: str, dsl: str, shortcut_id: str = "test") -> dict:
        return {
            "shortcut_id": shortcut_id,
            "messages": [
                {"role": "system", "content": "You are ShortcutForge..."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": dsl},
            ],
        }

    def test_label_simple_example(self):
        example = self._make_example(
            "Open the Calculator app",
            'SHORTCUT "Open Calc"\nACTION openapp WFAppIdentifier="com.apple.calculator"',
        )
        result = self.labeler.label_example(example)
        labels = result["labels"]
        self.assertEqual(labels["domain"], "general")
        self.assertEqual(labels["complexity"], "simple")
        self.assertEqual(labels["action_count"], 1)
        self.assertEqual(labels["construct_types"], [])
        self.assertGreater(labels["word_count"], 0)
        self.assertGreater(labels["dsl_char_len"], 0)

    def test_label_health_example(self):
        example = self._make_example(
            "Log my caffeine intake to HealthKit with the vitamin supplements",
            'SHORTCUT "Health Log"\nACTION health.quantity.log WFQuantitySampleType="Caffeine"\nACTION health.quantity.log WFQuantitySampleType="Vitamin C"',
        )
        result = self.labeler.label_example(example)
        labels = result["labels"]
        self.assertEqual(labels["domain"], "health_logger")
        self.assertEqual(labels["action_count"], 2)

    def test_label_api_example(self):
        example = self._make_example(
            "Fetch JSON data from the REST API endpoint and parse the response",
            'SHORTCUT "API Fetch"\nACTION url WFURLActionURL="https://api.example.com/data"\nACTION downloadurl WFHTTPMethod="GET"\nIF $response "is" ""\nACTION alert WFAlertActionTitle="Error"\nENDIF',
        )
        result = self.labeler.label_example(example)
        labels = result["labels"]
        self.assertEqual(labels["domain"], "api_workflow")
        self.assertIn("IF", labels["construct_types"])

    def test_label_file_example(self):
        example = self._make_example(
            "Save the document file to a folder in iCloud",
            'SHORTCUT "Save File"\nACTION savefile',
        )
        result = self.labeler.label_example(example)
        self.assertEqual(result["labels"]["domain"], "file_operations")

    def test_label_media_example(self):
        example = self._make_example(
            "Resize photos and crop the image to a thumbnail",
            'SHORTCUT "Resize"\nACTION resizeimage\nACTION cropimage',
        )
        result = self.labeler.label_example(example)
        self.assertEqual(result["labels"]["domain"], "media_processing")

    def test_label_scheduling_example(self):
        example = self._make_example(
            "Create a calendar event and set a reminder for the meeting",
            'SHORTCUT "Schedule"\nACTION addnewevent\nACTION addnewreminder',
        )
        result = self.labeler.label_example(example)
        self.assertEqual(result["labels"]["domain"], "scheduling")

    def test_label_messaging_example(self):
        example = self._make_example(
            "Send a text message to my contact via email",
            'SHORTCUT "Message"\nACTION sendmessage',
        )
        result = self.labeler.label_example(example)
        self.assertEqual(result["labels"]["domain"], "messaging")

    def test_label_malformed_example(self):
        example = {"messages": [{"role": "system", "content": "test"}]}
        result = self.labeler.label_example(example)
        self.assertEqual(result["labels"]["domain"], "unknown")

    def test_preserves_original_data(self):
        example = self._make_example("Test", 'SHORTCUT "T"\nACTION alert', "test-id")
        result = self.labeler.label_example(example)
        self.assertEqual(result["shortcut_id"], "test-id")
        self.assertEqual(len(result["messages"]), 3)

    def test_label_file_roundtrip(self):
        """Write examples to temp file, label them, verify output."""
        examples = [
            self._make_example("Open the app", 'SHORTCUT "T"\nACTION openapp'),
            self._make_example(
                "Set a timer for 5 minutes", 'SHORTCUT "T"\nACTION starttimer'
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            input_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        try:
            labeled = self.labeler.label_file(input_path, output_path)
            self.assertEqual(len(labeled), 2)

            # Verify output file
            with open(output_path) as f:
                output_lines = f.readlines()
            self.assertEqual(len(output_lines), 2)

            # Verify each line is valid JSON with labels
            for line in output_lines:
                data = json.loads(line)
                self.assertIn("labels", data)
                self.assertIn("domain", data["labels"])
                self.assertIn("complexity", data["labels"])
                self.assertIn("action_count", data["labels"])
                self.assertIn("construct_types", data["labels"])
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_label_file_skips_empty_lines(self):
        """Empty lines in JSONL should be skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(self._make_example("Test", 'SHORTCUT "T"\nACTION alert'))
                + "\n"
            )
            f.write("\n")
            f.write(
                json.dumps(self._make_example("Test2", 'SHORTCUT "T2"\nACTION alert'))
                + "\n"
            )
            input_path = Path(f.name)

        try:
            labeled = self.labeler.label_file(input_path)
            self.assertEqual(len(labeled), 2)
        finally:
            input_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
