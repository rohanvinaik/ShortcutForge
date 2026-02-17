"""
Tests for ContractValidator — 20 test cases covering all 13 rules.

Run: python -m pytest scripts/test_contract_validator.py -v
  or: python -m unittest scripts.test_contract_validator -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dsl_linter import lint_dsl
from dsl_parser import parse_dsl
from contract_validator import ContractValidator, ContractReport


def _parse(dsl: str):
    """Helper: lint + parse DSL text into ShortcutIR."""
    return parse_dsl(lint_dsl(dsl).text)


class TestContractValidator(unittest.TestCase):
    """Test suite for ContractValidator's 13 rules."""

    def setUp(self):
        self.validator = ContractValidator()

    # ── 1. Clean shortcut ────────────────────────────────────────

    def test_clean_shortcut_no_findings(self):
        """Simple clean shortcut produces no findings."""
        dsl = (
            'SHORTCUT "Clean"\n'
            'ACTION gettext Text="hello"\n'
            'SET $Msg = @prev\n'
            'IF $Msg has_any_value\n'
            '  ACTION alert WFAlertActionMessage=$Msg\n'
            'ENDIF\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        self.assertEqual(len(report.findings), 0)

    # ── 2. api.missing_error_check ───────────────────────────────

    def test_api_missing_error_check(self):
        """downloadurl without IF within 3 statements -> warning."""
        dsl = (
            'SHORTCUT "NoErrorCheck"\n'
            'ACTION url WFURLActionURL=$MyURL\n'
            'ACTION downloadurl\n'
            'ACTION alert WFAlertActionMessage="done"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "api.missing_error_check"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 3. api with error check ──────────────────────────────────

    def test_api_with_error_check_no_warning(self):
        """downloadurl followed by IF -> no missing_error_check warning."""
        dsl = (
            'SHORTCUT "WithErrorCheck"\n'
            'SET $URL = "https://example.com"\n'
            'ACTION url WFURLActionURL=$URL\n'
            'ACTION downloadurl\n'
            'SET $Response = @prev\n'
            'IF $Response has_any_value\n'
            '  ACTION alert WFAlertActionMessage="ok"\n'
            'ENDIF\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "api.missing_error_check"]
        self.assertEqual(len(rule_findings), 0)

    # ── 4. api.missing_content_type ──────────────────────────────

    def test_api_missing_content_type(self):
        """POST with body type but no headers -> warning."""
        dsl = (
            'SHORTCUT "NoContentType"\n'
            'SET $URL = "https://api.example.com"\n'
            'ACTION url WFURLActionURL=$URL\n'
            'ACTION downloadurl WFHTTPMethod="POST" WFHTTPBodyType="JSON"\n'
            'IF @prev has_any_value\n'
            '  ACTION alert WFAlertActionMessage="ok"\n'
            'ENDIF\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "api.missing_content_type"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 5. api.json_parse_after_fetch ────────────────────────────

    def test_api_json_parse_after_fetch(self):
        """downloadurl without detect.dictionary -> info."""
        dsl = (
            'SHORTCUT "NoParse"\n'
            'SET $URL = "https://api.example.com"\n'
            'ACTION url WFURLActionURL=$URL\n'
            'ACTION downloadurl\n'
            'SET $Data = @prev\n'
            'IF $Data has_any_value\n'
            '  ACTION alert WFAlertActionMessage="raw data"\n'
            'ENDIF\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "api.json_parse_after_fetch"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "info")

    # ── 6. api.url_not_variable ──────────────────────────────────

    def test_api_url_not_variable(self):
        """Hardcoded URL in downloadurl -> info."""
        dsl = (
            'SHORTCUT "HardcodedURL"\n'
            'ACTION downloadurl WFURLActionURL="https://api.example.com/data"\n'
            'IF @prev has_any_value\n'
            '  ACTION detect.dictionary\n'
            'ENDIF\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "api.url_not_variable"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "info")

    # ── 7. map.foreach_empty_body ────────────────────────────────

    def test_foreach_empty_body(self):
        """FOREACH with only 1 action in body -> warning."""
        dsl = (
            'SHORTCUT "EmptyForeach"\n'
            'SET $Items = "a"\n'
            'FOREACH $Items\n'
            '  ACTION alert WFAlertActionMessage="item"\n'
            'ENDFOREACH\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "map.foreach_empty_body"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 8. map.set_never_used ────────────────────────────────────

    def test_set_never_used(self):
        """SET $X but never referenced -> warning."""
        dsl = (
            'SHORTCUT "UnusedVar"\n'
            'SET $Unused = "hello"\n'
            'ACTION alert WFAlertActionMessage="bye"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "map.set_never_used"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")
        self.assertIn("Unused", rule_findings[0].message)

    # ── 9. SET used, no warning ──────────────────────────────────

    def test_set_used_no_warning(self):
        """SET $X then use $X in ACTION -> no set_never_used warning."""
        dsl = (
            'SHORTCUT "UsedVar"\n'
            'SET $Msg = "hello"\n'
            'ACTION alert WFAlertActionMessage=$Msg\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "map.set_never_used"]
        self.assertEqual(len(rule_findings), 0)

    # ── 10. Internal var skipped ─────────────────────────────────

    def test_internal_var_skipped(self):
        """SET $__internal never used -> no warning (skipped)."""
        dsl = (
            'SHORTCUT "InternalVar"\n'
            'SET $__internal = "temp"\n'
            'ACTION alert WFAlertActionMessage="hi"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "map.set_never_used"]
        # Should not have findings for __internal variables
        internal_findings = [f for f in rule_findings if "__internal" in f.message]
        self.assertEqual(len(internal_findings), 0)

    # ── 11. risk.infinite_repeat ─────────────────────────────────

    def test_infinite_repeat(self):
        """REPEAT 5000 -> warning."""
        dsl = (
            'SHORTCUT "BigRepeat"\n'
            'REPEAT 5000\n'
            '  ACTION alert WFAlertActionMessage="loop"\n'
            '  ACTION text Text="hi"\n'
            'ENDREPEAT\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "risk.infinite_repeat"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 12. Normal repeat, no warning ────────────────────────────

    def test_normal_repeat_no_warning(self):
        """REPEAT 10 -> no infinite_repeat warning."""
        dsl = (
            'SHORTCUT "SmallRepeat"\n'
            'REPEAT 10\n'
            '  ACTION alert WFAlertActionMessage="loop"\n'
            '  ACTION text Text="hi"\n'
            'ENDREPEAT\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "risk.infinite_repeat"]
        self.assertEqual(len(rule_findings), 0)

    # ── 13. risk.nested_network ──────────────────────────────────

    def test_nested_network(self):
        """downloadurl inside FOREACH -> warning."""
        dsl = (
            'SHORTCUT "NestedNet"\n'
            'SET $URLs = "items"\n'
            'FOREACH $URLs\n'
            '  ACTION downloadurl\n'
            '  ACTION detect.dictionary\n'
            'ENDFOREACH\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "risk.nested_network"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 14. risk.menu_duplicate_labels ───────────────────────────

    def test_menu_duplicate_labels(self):
        """MENU with two 'Option A' cases -> error."""
        dsl = (
            'SHORTCUT "DupMenu"\n'
            'MENU "Choose"\n'
            'CASE "Option A"\n'
            '  ACTION alert WFAlertActionMessage="A1"\n'
            'CASE "Option A"\n'
            '  ACTION alert WFAlertActionMessage="A2"\n'
            'CASE "Option B"\n'
            '  ACTION alert WFAlertActionMessage="B"\n'
            'ENDMENU\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "risk.menu_duplicate_labels"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "error")
        self.assertIn("Option A", rule_findings[0].message)

    # ── 15. Menu unique labels, no error ─────────────────────────

    def test_menu_unique_labels_no_error(self):
        """MENU with unique labels -> no duplicate_labels error."""
        dsl = (
            'SHORTCUT "UniqueMenu"\n'
            'MENU "Choose"\n'
            'CASE "Option A"\n'
            '  ACTION alert WFAlertActionMessage="A"\n'
            'CASE "Option B"\n'
            '  ACTION alert WFAlertActionMessage="B"\n'
            'CASE "Option C"\n'
            '  ACTION alert WFAlertActionMessage="C"\n'
            'ENDMENU\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "risk.menu_duplicate_labels"]
        self.assertEqual(len(rule_findings), 0)

    # ── 16. flow.use_before_set ──────────────────────────────────

    def test_use_before_set(self):
        """Reference $X before SET $X -> error."""
        dsl = (
            'SHORTCUT "UseBeforeSet"\n'
            'ACTION alert WFAlertActionMessage=$Greeting\n'
            'SET $Greeting = "hello"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "flow.use_before_set"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "error")
        self.assertIn("Greeting", rule_findings[0].message)

    # ── 17. flow.shadow_in_loop ──────────────────────────────────

    def test_shadow_in_loop(self):
        """SET $X outside, SET $X inside FOREACH -> warning."""
        dsl = (
            'SHORTCUT "Shadow"\n'
            'SET $Counter = 0\n'
            'SET $Items = "list"\n'
            'FOREACH $Items\n'
            '  SET $Counter = @prev\n'
            'ENDFOREACH\n'
            'ACTION alert WFAlertActionMessage=$Counter\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "flow.shadow_in_loop"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")
        self.assertIn("Counter", rule_findings[0].message)

    # ── 18. flow.dead_code_after_exit ────────────────────────────

    def test_dead_code_after_exit(self):
        """exitshortcut followed by more actions -> warning."""
        dsl = (
            'SHORTCUT "DeadCode"\n'
            'ACTION alert WFAlertActionMessage="before"\n'
            'ACTION exitshortcut\n'
            'ACTION alert WFAlertActionMessage="unreachable"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        rule_findings = [f for f in report.findings if f.rule_id == "flow.dead_code_after_exit"]
        self.assertTrue(len(rule_findings) >= 1)
        self.assertEqual(rule_findings[0].severity, "warning")

    # ── 19. Multiple findings ────────────────────────────────────

    def test_multiple_findings(self):
        """Shortcut with multiple issues -> multiple findings."""
        dsl = (
            'SHORTCUT "MultiIssue"\n'
            'ACTION alert WFAlertActionMessage=$Undefined\n'
            'SET $Unused = "never used"\n'
            'ACTION downloadurl WFURLActionURL="https://example.com"\n'
            'ACTION alert WFAlertActionMessage="done"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        # Should have at least:
        #   - flow.use_before_set for $Undefined
        #   - map.set_never_used for $Unused
        #   - api.missing_error_check for downloadurl without IF
        #   - api.url_not_variable for hardcoded URL
        self.assertTrue(len(report.findings) >= 3)
        rule_ids = {f.rule_id for f in report.findings}
        self.assertIn("flow.use_before_set", rule_ids)
        self.assertIn("map.set_never_used", rule_ids)
        self.assertIn("api.missing_error_check", rule_ids)

    # ── 20. Empty shortcut doesn't crash ─────────────────────────

    def test_validator_doesnt_crash_on_empty(self):
        """Empty shortcut -> no crash, no findings."""
        dsl = (
            'SHORTCUT "Empty"\n'
            'ENDSHORTCUT\n'
        )
        ir = _parse(dsl)
        report = self.validator.validate(ir)
        # Should not crash and should have no findings
        self.assertIsInstance(report, ContractReport)
        self.assertEqual(report.rules_checked, 13)
        self.assertFalse(report.has_errors)


if __name__ == "__main__":
    unittest.main()
