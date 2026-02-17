#!/usr/bin/env python3
"""Tests for macro_expander.py — Phase 1 macro expansion system.

Tests cover:
  - Registry loading and macro definitions
  - Parameter parsing (quoted, bare, $var, list, dict)
  - Template rendering (simple substitution, list iteration, defaults)
  - Block macro expansion (platform.if_ios etc.)
  - End-to-end expansion correctness
  - Idempotency (no macros → unchanged)
  - Unknown macro passthrough
  - Integration with lint pipeline
"""

import sys
import json
import tempfile
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from macro_expander import (
    MacroExpander,
    MacroExpansion,
    MacroDefinition,
    _parse_macro_params,
    _render_template,
    _load_registry,
    reload_registry,
    expand_macros,
    validate_macro_params,
)

# ── Test Harness ─────────────────────────────────────────────────────

_pass = 0
_fail = 0
_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    global _pass, _fail
    try:
        fn()
        _pass += 1
        _results.append((name, True, ""))
        print(f"  PASS: {name}")
    except Exception as e:
        _fail += 1
        _results.append((name, False, str(e)))
        print(f"  FAIL: {name} -- {e}")


# ── Registry Tests ──────────────────────────────────────────────────

def test_registry_loads():
    """Registry loads from macro_patterns.json and contains expected macros."""
    registry = reload_registry()
    assert len(registry) >= 7, f"Expected ≥7 macros, got {len(registry)}"
    assert "api.fetch_json" in registry
    assert "health.log_batch" in registry
    assert "platform.if_ios" in registry

run_test("registry_loads", test_registry_loads)


def test_registry_macro_fields():
    """Each macro definition has required fields."""
    registry = reload_registry()
    for name, defn in registry.items():
        assert defn.name == name, f"Name mismatch: {defn.name} != {name}"
        assert defn.description, f"Missing description for {name}"
        assert defn.expansion_template, f"Missing template for {name}"

run_test("registry_macro_fields", test_registry_macro_fields)


def test_registry_block_macros():
    """Block macros have end_marker and end_expansion."""
    registry = reload_registry()
    block_macros = {n: d for n, d in registry.items() if d.block_macro}
    assert len(block_macros) >= 2, f"Expected ≥2 block macros, got {len(block_macros)}"
    for name, defn in block_macros.items():
        assert defn.end_marker, f"Missing end_marker for block macro {name}"
        assert defn.end_expansion, f"Missing end_expansion for block macro {name}"

run_test("registry_block_macros", test_registry_block_macros)


def test_registry_custom_path():
    """Registry can load from a custom path."""
    # Create a minimal registry file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "macros": {
                "test.macro": {
                    "description": "Test macro",
                    "params": {},
                    "expansion_template": "ACTION comment WFCommentActionText=\"test\"",
                }
            }
        }, f)
        tmp_path = Path(f.name)

    try:
        registry = reload_registry(tmp_path)
        assert "test.macro" in registry
        assert registry["test.macro"].description == "Test macro"
    finally:
        tmp_path.unlink()
        # Reload the real registry
        reload_registry()

run_test("registry_custom_path", test_registry_custom_path)


# ── Parameter Parsing Tests ─────────────────────────────────────────

def test_parse_bare_params():
    """Parse bare word parameters."""
    params = _parse_macro_params('url=https://example.com method=GET')
    assert params["url"] == "https://example.com"
    assert params["method"] == "GET"

run_test("parse_bare_params", test_parse_bare_params)


def test_parse_quoted_params():
    """Parse double-quoted parameters."""
    params = _parse_macro_params('name="My API" desc="fetch data"')
    assert params["name"] == "My API"
    assert params["desc"] == "fetch data"

run_test("parse_quoted_params", test_parse_quoted_params)


def test_parse_var_ref_params():
    """Parse $variable reference parameters."""
    params = _parse_macro_params("source=$WeatherData target=$Output")
    assert params["source"] == "$WeatherData"
    assert params["target"] == "$Output"

run_test("parse_var_ref_params", test_parse_var_ref_params)


def test_parse_list_params():
    """Parse list parameters."""
    params = _parse_macro_params('items=["a","b","c"]')
    assert params["items"] == '["a","b","c"]'

run_test("parse_list_params", test_parse_list_params)


def test_parse_mixed_params():
    """Parse mixed parameter types."""
    params = _parse_macro_params('url=https://api.com source=$Data method="POST"')
    assert params["url"] == "https://api.com"
    assert params["source"] == "$Data"
    assert params["method"] == "POST"

run_test("parse_mixed_params", test_parse_mixed_params)


def test_parse_empty_params():
    """Parse empty parameter string returns empty dict."""
    params = _parse_macro_params("")
    assert params == {}

run_test("parse_empty_params", test_parse_empty_params)


# ── Template Rendering Tests ─────────────────────────────────────────

def test_render_simple_substitution():
    """Simple {{param}} substitution."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template="ACTION url WFURLActionURL=\"{{url}}\"",
    )
    result = _render_template(defn.expansion_template, {"url": "https://example.com"}, defn)
    assert "https://example.com" in result
    assert "{{url}}" not in result

run_test("render_simple_substitution", test_render_simple_substitution)


def test_render_multiple_params():
    """Multiple parameter substitution."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template="ACTION downloadurl WFHTTPMethod=\"{{method}}\" WFURLActionURL=\"{{url}}\"",
    )
    result = _render_template(
        defn.expansion_template,
        {"method": "POST", "url": "https://api.com"},
        defn,
    )
    assert "POST" in result
    assert "https://api.com" in result

run_test("render_multiple_params", test_render_multiple_params)


def test_render_default_values():
    """Default values from macro definition are applied."""
    defn = MacroDefinition(
        name="test", description="",
        params={"method": {"type": "string", "default": "GET"}},
        expansion_template="ACTION downloadurl WFHTTPMethod=\"{{method}}\"",
    )
    result = _render_template(defn.expansion_template, {}, defn)
    assert "GET" in result

run_test("render_default_values", test_render_default_values)


def test_render_list_iteration():
    """List iteration with {{#list}}...{{/list}}."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template='{{#items}}ACTION setclipboard Text="{{.}}"\n{{/items}}',
    )
    result = _render_template(
        defn.expansion_template,
        {"items": '["alpha", "beta", "gamma"]'},
        defn,
    )
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result

run_test("render_list_iteration", test_render_list_iteration)


# ── Expansion Tests ──────────────────────────────────────────────────

def test_expand_api_fetch_json():
    """api.fetch_json macro expands correctly."""
    text = 'MACRO api.fetch_json url=https://api.example.com/data'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert expansions[0].macro_name == "api.fetch_json"
    assert "url" in expanded.lower() or "downloadurl" in expanded.lower()
    assert "detect.dictionary" in expanded.lower()
    assert "MACRO" not in expanded  # Directive replaced

run_test("expand_api_fetch_json", test_expand_api_fetch_json)


def test_expand_health_log_batch():
    """health.log_batch macro expands correctly."""
    text = 'MACRO health.log_batch source=$NutrientData'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "health.quantity.log" in expanded.lower()
    assert "MACRO" not in expanded

run_test("expand_health_log_batch", test_expand_health_log_batch)


def test_expand_platform_block_macro():
    """platform.if_ios block macro expands with body content."""
    text = """MACRO platform.if_ios
  ACTION alert WFAlertActionMessage="iOS only!"
ENDPLATFORM"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "getdevicedetails" in expanded.lower()
    assert "alert" in expanded.lower()
    assert "ENDIF" in expanded
    assert "ENDPLATFORM" not in expanded
    assert "MACRO" not in expanded

run_test("expand_platform_block_macro", test_expand_platform_block_macro)


def test_expand_no_macros_unchanged():
    """Text without macros passes through unchanged."""
    text = """SHORTCUT "Test"
ACTION comment WFCommentActionText="hello"
ENDSHORTCUT"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 0
    assert expanded == text

run_test("expand_no_macros_unchanged", test_expand_no_macros_unchanged)


def test_expand_unknown_macro_passthrough():
    """Unknown macros are left as-is."""
    text = 'MACRO nonexistent.macro param=value'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 0
    assert "MACRO nonexistent.macro" in expanded

run_test("expand_unknown_macro_passthrough", test_expand_unknown_macro_passthrough)


def test_expand_case_insensitive():
    """MACRO directive matching is case-insensitive."""
    text = 'macro api.fetch_json url=https://example.com'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1

run_test("expand_case_insensitive", test_expand_case_insensitive)


def test_expand_multiple_macros():
    """Multiple macros in same text all expand."""
    text = """MACRO api.fetch_json url=https://api.com/a
ACTION comment WFCommentActionText="middle"
MACRO api.fetch_json url=https://api.com/b"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 2
    assert "middle" in expanded
    assert "MACRO" not in expanded

run_test("expand_multiple_macros", test_expand_multiple_macros)


def test_expand_mixed_with_dsl():
    """Macros expand correctly when mixed with regular DSL."""
    text = """SHORTCUT "Mixed"
ACTION comment WFCommentActionText="before"
MACRO api.fetch_json url=https://api.com/data
ACTION showresult Text="done"
ENDSHORTCUT"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert 'SHORTCUT "Mixed"' in expanded
    assert "ENDSHORTCUT" in expanded
    assert "before" in expanded
    assert "done" in expanded
    assert "MACRO" not in expanded

run_test("expand_mixed_with_dsl", test_expand_mixed_with_dsl)


def test_expansion_records_line_numbers():
    """Expansion records include correct line numbers."""
    text = """line1
line2
MACRO api.fetch_json url=https://example.com
line4"""
    expander = MacroExpander()
    _, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert expansions[0].line == 3  # 1-indexed

run_test("expansion_records_line_numbers", test_expansion_records_line_numbers)


def test_expansion_records_param_values():
    """Expansion records include parsed parameter values."""
    text = 'MACRO api.fetch_json url=https://example.com'
    expander = MacroExpander()
    _, expansions = expander.expand(text)

    assert expansions[0].param_values["url"] == "https://example.com"

run_test("expansion_records_param_values", test_expansion_records_param_values)


def test_convenience_function():
    """expand_macros() convenience function works."""
    text = 'MACRO api.fetch_json url=https://example.com'
    expanded, expansions = expand_macros(text)

    assert len(expansions) == 1
    assert "MACRO" not in expanded

run_test("convenience_function", test_convenience_function)


# ── Integration with Lint Pipeline ───────────────────────────────────

def test_lint_pipeline_expands_macros():
    """Macros are expanded in Phase 0 of lint pipeline."""
    from dsl_linter import lint_dsl

    text = """SHORTCUT "Lint Test"
MACRO api.fetch_json url=https://example.com
ENDSHORTCUT
"""
    result = lint_dsl(text)

    # Check that macro was expanded (no MACRO directive in output)
    assert "MACRO" not in result.text
    # Check that expansion was logged
    macro_changes = [c for c in result.changes if c.kind == "macro_expansion"]
    assert len(macro_changes) >= 1

run_test("lint_pipeline_expands_macros", test_lint_pipeline_expands_macros)


# ── Phase 5: Platform Branching Tests ──────────────────────────────

def test_platform_if_ios_expands():
    """IF_PLATFORM ios block macro expands to getdevicedetails + IF."""
    text = """MACRO platform.if_ios
  ACTION showresult WFText="iOS only!"
ENDPLATFORM"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "getdevicedetails" in expanded
    assert 'contains "iPhone"' in expanded
    assert "iOS only!" in expanded
    assert "ENDIF" in expanded
    assert "ENDPLATFORM" not in expanded
    assert "MACRO" not in expanded

run_test("platform_if_ios_expands", test_platform_if_ios_expands)


def test_platform_if_macos_expands():
    """IF_PLATFORM macos block macro expands correctly."""
    text = """MACRO platform.if_macos
  ACTION openapp WFAppName="Terminal"
ENDPLATFORM"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert "getdevicedetails" in expanded
    assert 'contains "Mac"' in expanded
    assert "Terminal" in expanded
    assert "ENDIF" in expanded

run_test("platform_if_macos_expands", test_platform_if_macos_expands)


def test_platform_if_watchos_expands():
    """IF_PLATFORM watchos block macro expands correctly."""
    text = """MACRO platform.if_watchos
  ACTION showresult WFText="Watch!"
ENDPLATFORM"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert "getdevicedetails" in expanded
    assert 'contains "Watch"' in expanded
    assert "Watch!" in expanded
    assert "ENDIF" in expanded

run_test("platform_if_watchos_expands", test_platform_if_watchos_expands)


def test_platform_block_parses_and_validates():
    """Expanded platform block parses and validates."""
    from dsl_linter import lint_dsl
    from dsl_parser import parse_dsl
    from dsl_validator import validate_ir

    text = """SHORTCUT "Platform Test"
MACRO platform.if_ios
  ACTION showresult WFText="Running on iOS"
ENDPLATFORM
ENDSHORTCUT
"""
    result = lint_dsl(text)
    ir = parse_dsl(result.text)
    assert ir is not None
    assert ir.action_count() >= 2  # getdevicedetails + showresult at minimum

    validation = validate_ir(ir, strict=False)
    assert len(validation.errors) == 0, (
        f"Validation errors: {[e.message for e in validation.errors]}"
    )

run_test("platform_block_parses_and_validates", test_platform_block_parses_and_validates)


def test_platform_multiple_blocks():
    """Multiple platform blocks in one shortcut expand correctly."""
    text = """MACRO platform.if_ios
  ACTION showresult WFText="iOS"
ENDPLATFORM
MACRO platform.if_macos
  ACTION showresult WFText="Mac"
ENDPLATFORM"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 2
    assert expanded.count("getdevicedetails") == 2
    assert expanded.count("ENDIF") == 2
    assert 'contains "iPhone"' in expanded
    assert 'contains "Mac"' in expanded

run_test("platform_multiple_blocks", test_platform_multiple_blocks)


# ── v2.0 Conditional Template Tests ──────────────────────────────────

def test_conditional_section_included():
    """{{?param}}...{{/param}} section included when param is present."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template='ACTION event Title="Meeting"{{?notes}} Notes="{{notes}}"{{/notes}}',
    )
    result = _render_template(defn.expansion_template, {"notes": "Agenda items"}, defn)
    assert 'Notes="Agenda items"' in result
    assert "{{?notes}}" not in result
    assert "{{/notes}}" not in result

run_test("conditional_section_included", test_conditional_section_included)


def test_conditional_section_omitted():
    """{{?param}}...{{/param}} section omitted when param is absent."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template='ACTION event Title="Meeting"{{?notes}} Notes="{{notes}}"{{/notes}}',
    )
    result = _render_template(defn.expansion_template, {}, defn)
    assert "Notes=" not in result
    assert "{{?notes}}" not in result
    assert "{{/notes}}" not in result
    assert 'Title="Meeting"' in result

run_test("conditional_section_omitted", test_conditional_section_omitted)


def test_conditional_multiple_sections():
    """Multiple conditional sections in one template expand independently."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template='ACTION event Title="T"{{?cal}} Cal="{{cal}}"{{/cal}}{{?notes}} Notes="{{notes}}"{{/notes}}',
    )
    # Provide only cal, not notes
    result = _render_template(defn.expansion_template, {"cal": "Work"}, defn)
    assert 'Cal="Work"' in result
    assert "Notes=" not in result

    # Provide both
    result2 = _render_template(defn.expansion_template, {"cal": "Work", "notes": "Hi"}, defn)
    assert 'Cal="Work"' in result2
    assert 'Notes="Hi"' in result2

run_test("conditional_multiple_sections", test_conditional_multiple_sections)


def test_conditional_with_list_iteration():
    """Conditional section alongside list iteration in same template."""
    defn = MacroDefinition(
        name="test", description="", params={},
        expansion_template='{{#items}}ITEM "{{.}}"\n{{/items}}{{?footer}}FOOTER "{{footer}}"{{/footer}}',
    )
    result = _render_template(
        defn.expansion_template,
        {"items": '["a","b"]', "footer": "done"},
        defn,
    )
    assert 'ITEM "a"' in result
    assert 'ITEM "b"' in result
    assert 'FOOTER "done"' in result

run_test("conditional_with_list_iteration", test_conditional_with_list_iteration)


def test_conditional_section_with_default_value():
    """Conditional section uses default value and is included when default applies."""
    defn = MacroDefinition(
        name="test", description="",
        params={"priority": {"type": "string", "default": "0"}},
        expansion_template='ACTION reminder Title="Task"{{?priority}} Priority={{priority}}{{/priority}}',
    )
    # No priority provided — default "0" should be applied, so conditional IS included
    result = _render_template(defn.expansion_template, {}, defn)
    assert "Priority=0" in result

run_test("conditional_section_with_default_value", test_conditional_section_with_default_value)


# ── v2.0 Parameter Validation Tests ─────────────────────────────────

def test_validate_missing_required_param():
    """Missing required parameter produces a warning."""
    defn = MacroDefinition(
        name="api.fetch_json", description="",
        params={"url": {"type": "string", "required": True}},
        expansion_template="ACTION url",
    )
    warnings = validate_macro_params("api.fetch_json", {}, defn)
    assert len(warnings) == 1
    assert "missing required" in warnings[0].lower()
    assert "url" in warnings[0]

run_test("validate_missing_required_param", test_validate_missing_required_param)


def test_validate_unknown_param():
    """Unknown parameter produces a warning."""
    defn = MacroDefinition(
        name="api.fetch_json", description="",
        params={"url": {"type": "string", "required": True}},
        expansion_template="ACTION url",
    )
    warnings = validate_macro_params("api.fetch_json", {"url": "x", "bogus": "y"}, defn)
    assert len(warnings) == 1
    assert "unknown parameter" in warnings[0].lower()
    assert "bogus" in warnings[0]

run_test("validate_unknown_param", test_validate_unknown_param)


def test_validate_valid_params_no_warnings():
    """Valid parameters produce no warnings."""
    defn = MacroDefinition(
        name="api.fetch_json", description="",
        params={
            "url": {"type": "string", "required": True},
            "method": {"type": "string", "required": False, "default": "GET"},
        },
        expansion_template="ACTION url",
    )
    warnings = validate_macro_params("api.fetch_json", {"url": "https://example.com"}, defn)
    assert warnings == []

run_test("validate_valid_params_no_warnings", test_validate_valid_params_no_warnings)


def test_validate_multiple_warnings():
    """Multiple validation issues produce multiple warnings."""
    defn = MacroDefinition(
        name="test", description="",
        params={
            "a": {"type": "string", "required": True},
            "b": {"type": "string", "required": True},
        },
        expansion_template="ACTION test",
    )
    # Missing both a and b, and providing unknown c
    warnings = validate_macro_params("test", {"c": "val"}, defn)
    assert len(warnings) == 3  # missing a + missing b + unknown c
    missing_warns = [w for w in warnings if "missing" in w.lower()]
    unknown_warns = [w for w in warnings if "unknown" in w.lower()]
    assert len(missing_warns) == 2
    assert len(unknown_warns) == 1

run_test("validate_multiple_warnings", test_validate_multiple_warnings)


def test_warnings_via_expander_property():
    """Validation warnings are accessible via MacroExpander.warnings property."""
    # Use a macro with a required param but don't provide it
    text = 'MACRO error.guard_input'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    # error.guard_input requires 'var' param — should produce a warning
    assert len(expander.warnings) >= 1
    assert any("var" in w for w in expander.warnings)

run_test("warnings_via_expander_property", test_warnings_via_expander_property)


# ── v2.0 New Macro Expansion Tests ──────────────────────────────────

# -- Error Handling --

def test_expand_error_guard_network():
    """error.guard_network expands to IF/ENDIF connectivity check pattern."""
    text = 'MACRO error.guard_network'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert expansions[0].macro_name == "error.guard_network"
    assert "getipaddress" in expanded.lower()
    assert "ENDIF" in expanded
    assert "exitshortcut" in expanded.lower()
    assert "Network Error" in expanded
    assert "MACRO" not in expanded

run_test("expand_error_guard_network", test_expand_error_guard_network)


def test_expand_error_guard_input_custom_message():
    """error.guard_input with a custom error message."""
    text = 'MACRO error.guard_input var=$UserName message="Please enter your name"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "$UserName" in expanded
    assert "Please enter your name" in expanded
    assert "ENDIF" in expanded
    assert "exitshortcut" in expanded.lower()

run_test("expand_error_guard_input_custom_message", test_expand_error_guard_input_custom_message)


def test_expand_error_retry_loop_block():
    """error.retry_loop block macro produces REPEAT start and end."""
    text = """MACRO error.retry_loop max_retries="5"
  ACTION downloadurl WFURLActionURL="https://api.com"
ENDRETRY"""
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "REPEAT 5" in expanded
    assert "ENDREPEAT" in expanded
    assert "$__retry_count" in expanded
    assert "downloadurl" in expanded.lower()
    assert "ENDRETRY" not in expanded
    assert "MACRO" not in expanded

run_test("expand_error_retry_loop_block", test_expand_error_retry_loop_block)


# -- API Contracts --

def test_expand_api_fetch_parse_guard():
    """api.fetch_parse_guard expands to fetch + guard + parse pipeline."""
    text = 'MACRO api.fetch_parse_guard url=https://api.example.com/data'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "https://api.example.com/data" in expanded
    assert "downloadurl" in expanded.lower()
    assert "exitshortcut" in expanded.lower()
    assert "detect.dictionary" in expanded.lower()
    assert "API Error" in expanded

run_test("expand_api_fetch_parse_guard", test_expand_api_fetch_parse_guard)


def test_expand_api_auth_header_bearer():
    """api.auth_header with default Bearer scheme."""
    text = 'MACRO api.auth_header token_var=$MyToken'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Bearer" in expanded
    assert "$MyToken" in expanded
    assert "$__auth_header" in expanded

run_test("expand_api_auth_header_bearer", test_expand_api_auth_header_bearer)


def test_expand_api_auth_header_basic():
    """api.auth_header with Basic scheme."""
    text = 'MACRO api.auth_header token_var=$Creds scheme="Basic"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Basic" in expanded
    assert "$Creds" in expanded
    assert "Bearer" not in expanded

run_test("expand_api_auth_header_basic", test_expand_api_auth_header_basic)


# -- File Operations --

def test_expand_file_save_with_picker():
    """file.save_with_picker expands to savefile action."""
    text = 'MACRO file.save_with_picker data_var=$Report'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "savefile" in expanded.lower()
    assert "$Report" in expanded
    assert "WFAskWhere=true" in expanded

run_test("expand_file_save_with_picker", test_expand_file_save_with_picker)


def test_expand_file_batch_rename():
    """file.batch_rename expands to FOREACH with rename logic."""
    text = 'MACRO file.batch_rename source_var=$Files pattern="draft" replacement="final"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "FOREACH" in expanded
    assert "$Files" in expanded
    assert "draft" in expanded
    assert "final" in expanded
    assert "replacetext" in expanded.lower()
    assert "rename" in expanded.lower()
    assert "ENDFOREACH" in expanded

run_test("expand_file_batch_rename", test_expand_file_batch_rename)


# -- Notifications --

def test_expand_notify_rich_alert_with_labels():
    """notify.rich_alert with explicit title and body."""
    text = 'MACRO notify.rich_alert title="Success" body="All items synced"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Success" in expanded
    assert "All items synced" in expanded
    assert "alert" in expanded.lower()
    assert "CancelButtonShown=true" in expanded

run_test("expand_notify_rich_alert_with_labels", test_expand_notify_rich_alert_with_labels)


def test_expand_notify_progress_update():
    """notify.progress_update expands to notification action."""
    text = 'MACRO notify.progress_update message="Processing step 3 of 5"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "notification" in expanded.lower()
    assert "Processing step 3 of 5" in expanded
    # Default subtitle should be applied
    assert "ShortcutForge" in expanded

run_test("expand_notify_progress_update", test_expand_notify_progress_update)


# -- Calendar --

def test_expand_calendar_create_event_minimal():
    """calendar.create_event with only required params (no optional calendar/notes)."""
    text = 'MACRO calendar.create_event title="Standup" start=$MeetingTime'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Standup" in expanded
    assert "$MeetingTime" in expanded
    assert "addnewevent" in expanded.lower()
    # Optional params not provided — conditional sections should be omitted
    assert 'WFCalendarItemCalendar' not in expanded
    assert 'WFCalendarItemNotes' not in expanded

run_test("expand_calendar_create_event_minimal", test_expand_calendar_create_event_minimal)


def test_expand_calendar_create_event_with_optionals():
    """calendar.create_event with optional calendar and notes."""
    text = 'MACRO calendar.create_event title="Review" start=$Date calendar="Work" notes="Q4 planning"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Review" in expanded
    assert 'WFCalendarItemCalendar="Work"' in expanded
    assert 'WFCalendarItemNotes="Q4 planning"' in expanded

run_test("expand_calendar_create_event_with_optionals", test_expand_calendar_create_event_with_optionals)


def test_expand_calendar_find_free_slot():
    """calendar.find_free_slot expands to calendar event query."""
    text = 'MACRO calendar.find_free_slot duration="45"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "getcalendarevents" in expanded.lower()
    assert "45" in expanded

run_test("expand_calendar_find_free_slot", test_expand_calendar_find_free_slot)


def test_expand_reminders_batch_create():
    """reminders.batch_create iterates via FOREACH over items."""
    text = 'MACRO reminders.batch_create items_var=$TodoItems list="Shopping"'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "FOREACH" in expanded
    assert "$TodoItems" in expanded
    assert "addnewreminder" in expanded.lower()
    assert 'WFReminderList="Shopping"' in expanded
    assert "ENDFOREACH" in expanded

run_test("expand_reminders_batch_create", test_expand_reminders_batch_create)


# -- Messaging --

def test_expand_message_send_with_confirm():
    """message.send_with_confirm expands to alert + sendmessage."""
    text = 'MACRO message.send_with_confirm recipient=$Contact body=$Msg'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "Confirm Send" in expanded
    assert "$Contact" in expanded
    assert "$Msg" in expanded
    assert "sendmessage" in expanded.lower()
    assert "alert" in expanded.lower()

run_test("expand_message_send_with_confirm", test_expand_message_send_with_confirm)


def test_expand_message_email_report():
    """message.email_report expands to sendemail action."""
    text = 'MACRO message.email_report to="team@co.com" subject="Weekly Report" body_var=$Report'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "team@co.com" in expanded
    assert "Weekly Report" in expanded
    assert "$Report" in expanded
    assert "sendemail" in expanded.lower()

run_test("expand_message_email_report", test_expand_message_email_report)


def test_expand_message_share_result():
    """message.share_result expands to share action."""
    text = 'MACRO message.share_result data_var=$Output'
    expander = MacroExpander()
    expanded, expansions = expander.expand(text)

    assert len(expansions) == 1
    assert "share" in expanded.lower()
    assert "$Output" in expanded

run_test("expand_message_share_result", test_expand_message_share_result)


# ── Report ───────────────────────────────────────────────────────────

print()
print("=" * 50)
print(f"Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
if _fail > 0:
    print(f"FAILURES: {_fail}")
    for name, passed, err in _results:
        if not passed:
            print(f"  {name}: {err}")
else:
    print("All tests passed.")
