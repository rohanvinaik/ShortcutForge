#!/usr/bin/env python3
"""
Tests for snippet_extractor.py.

Covers flattening, canonicalization, structural keys, scoring,
domain tagging, window extraction, deduplication, top-K selection,
registry format, round-trip query, and edge cases.

Run: python3 scripts/test_snippet_extractor.py
"""

import json
import sys
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))

from dsl_ir import (
    ShortcutIR,
    ActionStatement,
    SetVariable,
    IfBlock,
    MenuBlock,
    MenuCase,
    RepeatBlock,
    ForeachBlock,
    Comment,
    StringValue,
    NumberValue,
    BoolValue,
    VarRef,
    HandleRef,
    InterpolatedString,
)
from snippet_extractor import (
    SnippetExtractor,
    _flatten_ir,
    _canonicalize_variables,
    _compute_structural_key,
    _score_snippet,
    _detect_domain_tags,
    _has_control_flow,
    _generate_description,
    _reconstruct_canonical_dsl,
    FlatStatement,
    query_snippets,
    WINDOW_SIZES,
)


class TestFlattenIR(unittest.TestCase):
    """Test _flatten_ir correctly flattens nested IR."""

    def test_flat_actions_only(self):
        """Simple list of actions flattens 1:1."""
        stmts = [
            ActionStatement(action_name="gettext", params={}),
            ActionStatement(action_name="splittext", params={"WFInput": HandleRef("prev")}),
        ]
        flat = _flatten_ir(stmts)
        self.assertEqual(len(flat), 2)
        self.assertEqual(flat[0].kind, "action")
        self.assertEqual(flat[0].action_id, "gettext")
        self.assertEqual(flat[1].kind, "action")
        self.assertEqual(flat[1].action_id, "splittext")
        # Top-level actions are NOT in control flow
        self.assertFalse(flat[0].in_control_flow)
        self.assertFalse(flat[1].in_control_flow)

    def test_set_variable_flattened(self):
        """SET statements appear in the flat output."""
        stmts = [
            SetVariable(var_name="MyVar", value=HandleRef("prev")),
        ]
        flat = _flatten_ir(stmts)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0].kind, "set")
        self.assertEqual(flat[0].var_name, "MyVar")

    def test_nested_if_block(self):
        """IF block with then/else body produces boundary markers and nested stmts."""
        inner_action = ActionStatement(action_name="alert", params={"WFAlertActionMessage": StringValue("hello")})
        else_action = ActionStatement(action_name="showresult", params={})
        if_block = IfBlock(
            target=HandleRef("prev"),
            condition="has_any_value",
            compare_value=None,
            then_body=[inner_action],
            else_body=[else_action],
        )
        flat = _flatten_ir([if_block])

        # Expected: if_start, action(alert), else, action(showresult), endif
        self.assertEqual(len(flat), 5)
        self.assertEqual(flat[0].kind, "if_start")
        self.assertEqual(flat[1].kind, "action")
        self.assertEqual(flat[1].action_id, "alert")
        self.assertTrue(flat[1].in_control_flow)
        self.assertEqual(flat[2].kind, "else")
        self.assertEqual(flat[3].kind, "action")
        self.assertEqual(flat[3].action_id, "showresult")
        self.assertEqual(flat[4].kind, "endif")

    def test_foreach_block(self):
        """FOREACH produces foreach_start, body, endforeach."""
        body_action = ActionStatement(action_name="gettext", params={})
        foreach = ForeachBlock(
            collection=HandleRef("prev"),
            body=[body_action],
        )
        flat = _flatten_ir([foreach])
        self.assertEqual(len(flat), 3)
        self.assertEqual(flat[0].kind, "foreach_start")
        self.assertEqual(flat[1].kind, "action")
        self.assertTrue(flat[1].in_control_flow)
        self.assertEqual(flat[2].kind, "endforeach")

    def test_deeply_nested(self):
        """Deeply nested structure (FOREACH containing IF) flattens correctly."""
        inner_action = ActionStatement(action_name="notification", params={})
        if_block = IfBlock(
            target=HandleRef("item"),
            condition="has_any_value",
            compare_value=None,
            then_body=[inner_action],
            else_body=None,
        )
        foreach = ForeachBlock(
            collection=HandleRef("prev"),
            body=[if_block],
        )
        flat = _flatten_ir([foreach])
        # foreach_start, if_start, action, endif, endforeach
        self.assertEqual(len(flat), 5)
        kinds = [f.kind for f in flat]
        self.assertEqual(kinds, ["foreach_start", "if_start", "action", "endif", "endforeach"])
        # The inner action should be in_control_flow
        self.assertTrue(flat[2].in_control_flow)

    def test_comment_included(self):
        """Comments appear as flat statements."""
        stmts = [
            Comment(text="A note"),
            ActionStatement(action_name="gettext", params={}),
        ]
        flat = _flatten_ir(stmts)
        self.assertEqual(len(flat), 2)
        self.assertEqual(flat[0].kind, "comment")

    def test_action_name_normalization(self):
        """is.workflow.actions. prefix is stripped in action_id."""
        stmts = [
            ActionStatement(action_name="is.workflow.actions.downloadurl", params={}),
        ]
        flat = _flatten_ir(stmts)
        self.assertEqual(flat[0].action_id, "downloadurl")


class TestCanonicalizeVariables(unittest.TestCase):
    """Test _canonicalize_variables replaces vars in order."""

    def test_basic_renaming(self):
        """Variables are renamed in first-appearance order."""
        flat = [
            FlatStatement(kind="set", var_name="UserInput", value_repr="@prev"),
            FlatStatement(kind="action", action_id="gettext",
                          params={"WFInput": "$UserInput"}),
            FlatStatement(kind="set", var_name="Result", value_repr="@prev"),
        ]
        canonical = _canonicalize_variables(flat)
        self.assertEqual(canonical[0].var_name, "__v1")
        self.assertEqual(canonical[1].params["WFInput"], "$__v1")
        self.assertEqual(canonical[2].var_name, "__v2")

    def test_handles_not_renamed(self):
        """Handle references like @prev are NOT renamed."""
        flat = [
            FlatStatement(kind="action", action_id="gettext",
                          params={"WFInput": "@prev"}),
        ]
        canonical = _canonicalize_variables(flat)
        self.assertEqual(canonical[0].params["WFInput"], "@prev")

    def test_same_var_gets_same_canonical(self):
        """Multiple references to the same variable get the same canonical name."""
        flat = [
            FlatStatement(kind="set", var_name="X", value_repr="@prev"),
            FlatStatement(kind="action", action_id="alert",
                          params={"Msg": "$X"}),
            FlatStatement(kind="action", action_id="showresult",
                          params={"Text": "$X"}),
        ]
        canonical = _canonicalize_variables(flat)
        self.assertEqual(canonical[0].var_name, "__v1")
        self.assertEqual(canonical[1].params["Msg"], "$__v1")
        self.assertEqual(canonical[2].params["Text"], "$__v1")


class TestComputeStructuralKey(unittest.TestCase):
    """Test _compute_structural_key produces dot-joined action IDs."""

    def test_basic_key(self):
        flat = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="action", action_id="splittext"),
            FlatStatement(kind="action", action_id="count"),
        ]
        key = _compute_structural_key(flat)
        self.assertEqual(key, "gettext.splittext.count")

    def test_set_excluded(self):
        """SET statements do not contribute to the structural key."""
        flat = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="set", var_name="X"),
            FlatStatement(kind="action", action_id="splittext"),
        ]
        key = _compute_structural_key(flat)
        self.assertEqual(key, "gettext.splittext")

    def test_control_flow_markers_excluded(self):
        """Control flow markers (if_start, etc.) do not appear in key."""
        flat = [
            FlatStatement(kind="action", action_id="downloadurl"),
            FlatStatement(kind="if_start"),
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="endif"),
        ]
        key = _compute_structural_key(flat)
        self.assertEqual(key, "downloadurl.alert")


class TestScoreSnippet(unittest.TestCase):
    """Test _score_snippet scoring formula."""

    def test_high_diversity(self):
        """All unique actions should have diversity_ratio = 1.0."""
        action_ids = ["gettext", "splittext", "count"]
        score = _score_snippet(frequency=5, action_ids=action_ids, has_cf=False)
        # 5 * (3/3) * 1.0 = 5.0
        self.assertAlmostEqual(score, 5.0)

    def test_low_diversity(self):
        """Repeated actions reduce diversity ratio."""
        action_ids = ["gettext", "gettext", "gettext", "splittext"]
        score = _score_snippet(frequency=4, action_ids=action_ids, has_cf=False)
        # 4 * (2/4) * 1.0 = 2.0
        self.assertAlmostEqual(score, 2.0)

    def test_control_flow_bonus(self):
        """Control flow adds 0.5 weight multiplier."""
        action_ids = ["downloadurl", "alert"]
        score_no_cf = _score_snippet(frequency=3, action_ids=action_ids, has_cf=False)
        score_with_cf = _score_snippet(frequency=3, action_ids=action_ids, has_cf=True)
        # Without CF: 3 * 1.0 * 1.0 = 3.0
        # With CF: 3 * 1.0 * 1.5 = 4.5
        self.assertAlmostEqual(score_no_cf, 3.0)
        self.assertAlmostEqual(score_with_cf, 4.5)

    def test_empty_actions(self):
        """Empty action list returns score 0."""
        score = _score_snippet(frequency=10, action_ids=[], has_cf=False)
        self.assertEqual(score, 0.0)


class TestDetectDomainTags(unittest.TestCase):
    """Test _detect_domain_tags correct domain assignment."""

    def test_text_processing(self):
        tags = _detect_domain_tags(["gettext", "splittext"])
        self.assertIn("text_processing", tags)

    def test_networking(self):
        tags = _detect_domain_tags(["downloadurl", "geturlcomponent"])
        self.assertIn("networking", tags)

    def test_health(self):
        tags = _detect_domain_tags(["health.quantity.log"])
        self.assertIn("health", tags)

    def test_multiple_domains(self):
        tags = _detect_domain_tags(["downloadurl", "splittext", "alert"])
        self.assertIn("networking", tags)
        self.assertIn("text_processing", tags)
        self.assertIn("user_feedback", tags)

    def test_general_fallback(self):
        """Unknown actions get the 'general' tag."""
        tags = _detect_domain_tags(["someunknownaction"])
        self.assertEqual(tags, ["general"])

    def test_messaging(self):
        tags = _detect_domain_tags(["sendmessage", "sendemail"])
        self.assertIn("messaging", tags)

    def test_scheduling(self):
        tags = _detect_domain_tags(["addnewevent", "addnewreminder"])
        self.assertIn("scheduling", tags)

    def test_media(self):
        tags = _detect_domain_tags(["image.resize", "image.crop"])
        self.assertIn("media", tags)


class TestWindowExtraction(unittest.TestCase):
    """Test sliding window extraction with various sizes."""

    def _make_extractor_with_ir(self, stmts: list, top_k: int = 200) -> SnippetExtractor:
        """Helper: create extractor and feed it an IR directly."""
        ext = SnippetExtractor(top_k=top_k)
        flat = _flatten_ir(stmts)
        for wsize in WINDOW_SIZES:
            if wsize > len(flat):
                continue
            for start in range(len(flat) - wsize + 1):
                window = flat[start:start + wsize]
                ext._process_window(window)
        ext._total_examples = 1
        return ext

    def test_window_size_3(self):
        """Window size 3 extracts 3-action sub-sequences."""
        stmts = [
            ActionStatement(action_name="gettext", params={}),
            ActionStatement(action_name="splittext", params={}),
            ActionStatement(action_name="count", params={}),
            ActionStatement(action_name="alert", params={}),
        ]
        ext = self._make_extractor_with_ir(stmts)
        # Should have windows of size 3: (gettext,splittext,count), (splittext,count,alert)
        keys = list(ext._key_candidates.keys())
        self.assertIn("gettext.splittext.count", keys)
        self.assertIn("splittext.count.alert", keys)

    def test_window_size_5(self):
        """Window size 5 captures longer sequences."""
        stmts = [
            ActionStatement(action_name="downloadurl", params={}),
            ActionStatement(action_name="gettext", params={}),
            ActionStatement(action_name="splittext", params={}),
            ActionStatement(action_name="count", params={}),
            ActionStatement(action_name="alert", params={}),
        ]
        ext = self._make_extractor_with_ir(stmts)
        self.assertIn("downloadurl.gettext.splittext.count.alert",
                       list(ext._key_candidates.keys()))

    def test_short_sequence_skipped(self):
        """Sequences shorter than smallest window size (3) produce no windows."""
        stmts = [
            ActionStatement(action_name="gettext", params={}),
            ActionStatement(action_name="alert", params={}),
        ]
        ext = self._make_extractor_with_ir(stmts)
        self.assertEqual(len(ext._key_candidates), 0)


class TestDeduplication(unittest.TestCase):
    """Test deduplication by structural key."""

    def test_same_key_counted(self):
        """Two windows with the same structural key increment the count."""
        ext = SnippetExtractor()

        # Feed the same pattern twice
        window1 = [
            FlatStatement(kind="action", action_id="gettext", params={"WFInput": "$A"}),
            FlatStatement(kind="action", action_id="splittext", params={}),
            FlatStatement(kind="action", action_id="count", params={}),
        ]
        window2 = [
            FlatStatement(kind="action", action_id="gettext", params={"WFInput": "$B"}),
            FlatStatement(kind="action", action_id="splittext", params={}),
            FlatStatement(kind="action", action_id="count", params={}),
        ]
        ext._process_window(window1)
        ext._process_window(window2)

        self.assertEqual(ext._key_counts["gettext.splittext.count"], 2)
        # Only one candidate entry despite two occurrences
        self.assertEqual(len(ext._key_candidates), 1)


class TestTopKSelection(unittest.TestCase):
    """Test top-K selection returns correct number of results."""

    def test_top_k_limits_output(self):
        """extract() returns at most top_k snippets."""
        ext = SnippetExtractor(top_k=2)

        # Create 3 distinct candidates
        for i, aid_pair in enumerate([
            ("gettext", "splittext", "count"),
            ("downloadurl", "alert", "showresult"),
            ("sendmessage", "notification", "exit"),
        ]):
            window = [
                FlatStatement(kind="action", action_id=aid_pair[0]),
                FlatStatement(kind="action", action_id=aid_pair[1]),
                FlatStatement(kind="action", action_id=aid_pair[2]),
            ]
            # Give different frequencies for deterministic ordering
            for _ in range(i + 1):
                ext._process_window(window)

        snippets = ext.extract()
        self.assertEqual(len(snippets), 2)
        # Highest-frequency snippet should come first
        self.assertGreaterEqual(snippets[0]["score"], snippets[1]["score"])


class TestRegistryFormat(unittest.TestCase):
    """Test registry output format validation."""

    def test_registry_structure(self):
        """Registry dict has required top-level keys and snippet format."""
        ext = SnippetExtractor(top_k=10)

        window = [
            FlatStatement(kind="action", action_id="downloadurl"),
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="action", action_id="splittext"),
        ]
        ext._process_window(window)

        registry = ext.build_registry("test_input.jsonl")

        # Top-level keys
        self.assertEqual(registry["version"], "1.0")
        self.assertEqual(registry["extracted_from"], "test_input.jsonl")
        self.assertIn("snippet_count", registry)
        self.assertIn("snippets", registry)
        self.assertEqual(registry["snippet_count"], len(registry["snippets"]))

        # Snippet format
        snip = registry["snippets"][0]
        self.assertIn("id", snip)
        self.assertTrue(snip["id"].startswith("snip_"))
        self.assertIn("structural_key", snip)
        self.assertIn("canonical_dsl", snip)
        self.assertIn("action_count", snip)
        self.assertIn("frequency", snip)
        self.assertIn("score", snip)
        self.assertIn("domain_tags", snip)
        self.assertIn("description", snip)
        self.assertIsInstance(snip["domain_tags"], list)
        self.assertIsInstance(snip["score"], float)


class TestRoundTrip(unittest.TestCase):
    """Test extract -> save -> load -> query round-trip."""

    def test_extract_load_query(self):
        """Registry written to disk can be loaded and queried."""
        ext = SnippetExtractor(top_k=10)

        # Add a text-processing snippet
        window = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="action", action_id="splittext"),
            FlatStatement(kind="action", action_id="count"),
        ]
        ext._process_window(window)

        # Add a networking snippet
        window2 = [
            FlatStatement(kind="action", action_id="downloadurl"),
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="action", action_id="alert"),
        ]
        ext._process_window(window2)

        registry = ext.build_registry("test.jsonl")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump(registry, f)
            tmp_path = Path(f.name)

        try:
            results = query_snippets("split text into lines", registry_path=tmp_path, top_k=3)
            self.assertGreater(len(results), 0)
            # The text-processing snippet should match "split"
            keys = [r["structural_key"] for r in results]
            self.assertTrue(
                any("splittext" in k for k in keys),
                f"Expected a splittext snippet in results, got keys: {keys}"
            )
        finally:
            tmp_path.unlink()


class TestQueryFunction(unittest.TestCase):
    """Test query_snippets function."""

    def _make_registry(self, snippets: list[dict]) -> Path:
        """Write a temporary registry and return its path."""
        registry = {
            "version": "1.0",
            "extracted_from": "test.jsonl",
            "snippet_count": len(snippets),
            "snippets": snippets,
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(registry, f)
        f.close()
        return Path(f.name)

    def test_returns_relevant_results(self):
        """Query with 'download' keyword returns networking snippet."""
        path = self._make_registry([
            {
                "id": "snip_001",
                "structural_key": "downloadurl.gettext.alert",
                "canonical_dsl": "ACTION downloadurl\nACTION gettext\nACTION alert",
                "action_count": 3,
                "frequency": 5,
                "score": 5.0,
                "domain_tags": ["networking", "text_processing", "user_feedback"],
                "description": "Fetch URL, get text, show alert",
            },
            {
                "id": "snip_002",
                "structural_key": "sendmessage.notification",
                "canonical_dsl": "ACTION sendmessage\nACTION notification",
                "action_count": 2,
                "frequency": 3,
                "score": 3.0,
                "domain_tags": ["messaging"],
                "description": "Send message, send notification",
            },
        ])
        try:
            results = query_snippets("download a file from the web", registry_path=path)
            self.assertGreater(len(results), 0)
            # The networking snippet should rank first
            self.assertIn("downloadurl", results[0]["structural_key"])
        finally:
            path.unlink()

    def test_handles_empty_registry(self):
        """Query on empty registry returns empty list."""
        path = self._make_registry([])
        try:
            results = query_snippets("do something", registry_path=path)
            self.assertEqual(results, [])
        finally:
            path.unlink()

    def test_missing_registry_returns_empty(self):
        """Non-existent registry path returns empty list."""
        results = query_snippets(
            "anything",
            registry_path=Path("/nonexistent/path/registry.json"),
        )
        self.assertEqual(results, [])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_single_action_window_skipped(self):
        """A window with only 1 unique action ID is skipped (< 2 unique)."""
        ext = SnippetExtractor()
        window = [
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="set", var_name="X", value_repr="@prev"),
        ]
        ext._process_window(window)
        # Only 1 unique action ID -> should be skipped
        self.assertEqual(len(ext._key_candidates), 0)

    def test_all_same_actions_low_score(self):
        """All same actions have low diversity ratio = 1/N."""
        # 3 identical actions: diversity = 1/3
        action_ids = ["alert", "alert", "alert"]
        score = _score_snippet(frequency=10, action_ids=action_ids, has_cf=False)
        # 10 * (1/3) * 1.0 ~ 3.333
        self.assertAlmostEqual(score, 10 / 3, places=3)
        # Compare to 3 unique actions
        diverse_score = _score_snippet(frequency=10, action_ids=["a", "b", "c"], has_cf=False)
        # 10 * 1.0 * 1.0 = 10.0
        self.assertAlmostEqual(diverse_score, 10.0)
        self.assertGreater(diverse_score, score)

    def test_all_same_actions_skipped_by_extractor(self):
        """Window with all identical action IDs has < 2 unique -> skipped."""
        ext = SnippetExtractor()
        window = [
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="action", action_id="alert"),
        ]
        ext._process_window(window)
        # < 2 unique action IDs -> skipped
        self.assertEqual(len(ext._key_candidates), 0)


class TestDescriptionGeneration(unittest.TestCase):
    """Test auto-generation of descriptions."""

    def test_known_actions_readable(self):
        """Known actions produce human-readable descriptions."""
        desc = _generate_description(["downloadurl", "gettext", "alert"], has_cf=False)
        self.assertIn("Fetch URL", desc)
        self.assertIn("Get text", desc)
        self.assertIn("Show alert", desc)

    def test_unknown_action_uses_id(self):
        """Unknown actions use the action_id as description."""
        desc = _generate_description(["some.custom.action"], has_cf=False)
        # Should contain the action id with dots replaced
        self.assertIn("some", desc.lower())

    def test_control_flow_noted(self):
        """Description mentions control flow when present."""
        desc = _generate_description(["gettext", "splittext"], has_cf=True)
        self.assertIn("control flow", desc.lower())


class TestHasControlFlow(unittest.TestCase):
    """Test _has_control_flow detection."""

    def test_with_if(self):
        flat = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="if_start"),
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="endif"),
        ]
        self.assertTrue(_has_control_flow(flat))

    def test_without_cf(self):
        flat = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="action", action_id="splittext"),
        ]
        self.assertFalse(_has_control_flow(flat))

    def test_with_foreach(self):
        flat = [
            FlatStatement(kind="foreach_start"),
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="endforeach"),
        ]
        self.assertTrue(_has_control_flow(flat))


class TestIngestDSL(unittest.TestCase):
    """Test full DSL ingestion via SnippetExtractor.ingest_dsl."""

    def test_ingest_valid_dsl(self):
        """A valid multi-action DSL produces snippet candidates."""
        dsl = (
            'SHORTCUT "Test Snippet"\n'
            'ACTION downloadurl URL="https://example.com"\n'
            'SET $Response = @prev\n'
            'ACTION gettext WFInput=$Response\n'
            'ACTION splittext WFInput=@prev\n'
            'ACTION count Input=@prev\n'
            'ACTION alert WFAlertActionMessage="Done"\n'
            'ENDSHORTCUT\n'
        )
        ext = SnippetExtractor()
        ext.ingest_dsl(dsl)

        self.assertEqual(ext._total_examples, 1)
        self.assertEqual(ext._parse_errors, 0)
        self.assertGreater(len(ext._key_candidates), 0)

    def test_ingest_invalid_dsl_counted_as_error(self):
        """Invalid DSL is counted as parse error without crashing."""
        ext = SnippetExtractor()
        ext.ingest_dsl("THIS IS NOT VALID DSL AT ALL")
        self.assertEqual(ext._total_examples, 1)
        self.assertEqual(ext._parse_errors, 1)
        self.assertEqual(len(ext._key_candidates), 0)


class TestReconstructCanonicalDSL(unittest.TestCase):
    """Test _reconstruct_canonical_dsl produces valid DSL-like output."""

    def test_basic_reconstruction(self):
        """Simple action sequence reconstructs to ACTION lines."""
        flat = [
            FlatStatement(kind="action", action_id="gettext", params={"WFInput": "@prev"}),
            FlatStatement(kind="action", action_id="splittext", params={}),
        ]
        dsl = _reconstruct_canonical_dsl(flat)
        self.assertIn("ACTION gettext", dsl)
        self.assertIn("ACTION splittext", dsl)

    def test_control_flow_reconstruction(self):
        """Control flow markers produce IF/ENDIF/FOREACH/etc."""
        flat = [
            FlatStatement(kind="action", action_id="gettext"),
            FlatStatement(kind="if_start"),
            FlatStatement(kind="action", action_id="alert"),
            FlatStatement(kind="endif"),
        ]
        dsl = _reconstruct_canonical_dsl(flat)
        self.assertIn("IF", dsl)
        self.assertIn("ENDIF", dsl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
