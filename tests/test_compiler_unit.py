"""
Unit tests for the Apple Shortcuts compiler.
Tests bug fixes and core invariants.

Run: python3 scripts/test_compiler_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from shortcuts_compiler import CONDITION_MAP, Shortcut, _resolve_identifier, actions

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {name} — {e}")


# ============================================================
# Phase 1A: bool/int type detection
# ============================================================


def test_bool_before_int():
    items = actions.build_dict_items(
        {
            "flag": True,
            "disabled": False,
            "count": 42,
            "rate": 3.14,
            "name": "test",
        }
    )
    type_map = {item["WFKey"]["Value"]["string"]: item["WFItemType"] for item in items}
    assert type_map["flag"] == 6, (
        f"True should be type 6 (bool), got {type_map['flag']}"
    )
    assert type_map["disabled"] == 6, (
        f"False should be type 6 (bool), got {type_map['disabled']}"
    )
    assert type_map["count"] == 3, (
        f"int should be type 3 (number), got {type_map['count']}"
    )
    assert type_map["rate"] == 3, (
        f"float should be type 3 (number), got {type_map['rate']}"
    )
    assert type_map["name"] == 0, (
        f"str should be type 0 (string), got {type_map['name']}"
    )


def test_dict_and_list_types():
    items = actions.build_dict_items(
        {
            "nested": {"a": 1},
            "items": [1, 2, 3],
        }
    )
    type_map = {item["WFKey"]["Value"]["string"]: item["WFItemType"] for item in items}
    assert type_map["nested"] == 1, f"dict should be type 1, got {type_map['nested']}"
    assert type_map["items"] == 2, f"list should be type 2, got {type_map['items']}"


# ============================================================
# Action resolution
# ============================================================


def test_resolve_short_name():
    assert _resolve_identifier("comment") == "is.workflow.actions.comment"


def test_resolve_dotted_name():
    assert _resolve_identifier("text.split") == "is.workflow.actions.text.split"


def test_resolve_full_identifier():
    assert (
        _resolve_identifier("is.workflow.actions.gettext")
        == "is.workflow.actions.gettext"
    )


def test_resolve_third_party():
    # Third-party identifiers starting with com. should pass through
    ident = "com.apple.Pages.TSADocumentCreateIntent"
    assert _resolve_identifier(ident) == ident


def test_resolve_unknown_raises():
    try:
        _resolve_identifier("completely_nonexistent_action_xyz_12345")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ============================================================
# Parameter validation
# ============================================================


def test_make_rejects_unknown_params():
    try:
        actions.make("comment", WFCommentActionText="ok", TotallyFakeParam="bad")
        assert False, "Should have raised ValueError for unknown param"
    except ValueError as e:
        assert "Unknown parameter" in str(e)


def test_make_accepts_valid_params():
    action = actions.make("comment", WFCommentActionText="Hello")
    assert action["WFWorkflowActionIdentifier"] == "is.workflow.actions.comment"
    assert action["WFWorkflowActionParameters"]["WFCommentActionText"] == "Hello"


# ============================================================
# Control flow structure
# ============================================================


def test_if_block_structure():
    s = Shortcut("Test")
    handle = s.add(
        actions.make("gettext", WFTextActionText=actions.build_token_string("hello"))
    )
    with s.if_block(handle, condition="has_any_value"):
        s.add(actions.make("comment", WFCommentActionText="inside if"))

    conditionals = [
        a for a in s.actions if "conditional" in a.get("WFWorkflowActionIdentifier", "")
    ]
    assert len(conditionals) == 2, (
        f"Expected 2 conditionals (start+end), got {len(conditionals)}"
    )
    assert conditionals[0]["WFWorkflowActionParameters"]["WFControlFlowMode"] == 0
    assert conditionals[1]["WFWorkflowActionParameters"]["WFControlFlowMode"] == 2
    gid_start = conditionals[0]["WFWorkflowActionParameters"]["GroupingIdentifier"]
    gid_end = conditionals[1]["WFWorkflowActionParameters"]["GroupingIdentifier"]
    assert gid_start == gid_end, "GroupingIdentifiers must match"


def test_if_else_block_structure():
    s = Shortcut("Test")
    handle = s.add(
        actions.make("gettext", WFTextActionText=actions.build_token_string("hello"))
    )
    with s.if_else_block(handle, condition="has_any_value") as otherwise:
        s.add(actions.make("comment", WFCommentActionText="if branch"))
        otherwise()
        s.add(actions.make("comment", WFCommentActionText="else branch"))

    conditionals = [
        a for a in s.actions if "conditional" in a.get("WFWorkflowActionIdentifier", "")
    ]
    assert len(conditionals) == 3, (
        f"Expected 3 conditionals (start+else+end), got {len(conditionals)}"
    )
    modes = [c["WFWorkflowActionParameters"]["WFControlFlowMode"] for c in conditionals]
    assert modes == [0, 1, 2], f"Expected modes [0, 1, 2], got {modes}"
    gids = [c["WFWorkflowActionParameters"]["GroupingIdentifier"] for c in conditionals]
    assert len(set(gids)) == 1, "All GroupingIdentifiers must match"


def test_menu_block_structure():
    s = Shortcut("Test")
    with s.menu_block("Choose:", ["A", "B", "C"]) as cases:
        cases["A"]()
        s.add(actions.make("comment", WFCommentActionText="A"))
        cases["B"]()
        s.add(actions.make("comment", WFCommentActionText="B"))
        cases["C"]()
        s.add(actions.make("comment", WFCommentActionText="C"))

    menus = [
        a
        for a in s.actions
        if "choosefrommenu" in a.get("WFWorkflowActionIdentifier", "")
    ]
    assert len(menus) == 5, (
        f"Expected 5 menu actions (start+3 cases+end), got {len(menus)}"
    )
    modes = [m["WFWorkflowActionParameters"]["WFControlFlowMode"] for m in menus]
    assert modes == [0, 1, 1, 1, 2], f"Expected modes [0,1,1,1,2], got {modes}"


def test_nested_control_flow():
    """Verify nested if inside menu inside repeat produces valid structure."""
    s = Shortcut("Test")
    items = s.add(actions.make("list", WFItems=actions.build_list(["a", "b"])))
    with s.repeat_each_block(items):
        with s.menu_block("Pick:", ["X", "Y"]) as cases:
            cases["X"]()
            handle = s.add(
                actions.make(
                    "gettext", WFTextActionText=actions.build_token_string("test")
                )
            )
            with s.if_block(handle, condition="has_any_value"):
                s.add(actions.make("comment", WFCommentActionText="nested"))
            cases["Y"]()
            s.add(actions.make("comment", WFCommentActionText="Y"))

    warnings = s.validate()
    crash_warnings = [w for w in warnings if "crash" in w.lower()]
    assert not crash_warnings, (
        f"Nested control flow produced crash warnings: {crash_warnings}"
    )


# ============================================================
# Builder helpers
# ============================================================


def test_build_list():
    result = actions.build_list(["a", "b", "c"])
    assert len(result) == 3
    assert all(item["WFItemType"] == 0 for item in result)
    assert result[0]["WFValue"]["Value"]["string"] == "a"


def test_build_headers():
    result = actions.build_headers({"Authorization": "Bearer xyz"})
    assert "Value" in result
    items = result["Value"]["WFDictionaryFieldValueItems"]
    assert len(items) == 1
    assert items[0]["WFKey"]["Value"]["string"] == "Authorization"


def test_build_quantity():
    result = actions.build_quantity(7, "days")
    assert result["WFSerializationType"] == "WFQuantityFieldValue"
    assert result["Value"]["Magnitude"] == 7
    assert result["Value"]["Unit"] == "days"


def test_build_token_string():
    result = actions.build_token_string("Hello world")
    assert result["WFSerializationType"] == "WFTextTokenString"
    assert result["Value"]["string"] == "Hello world"


# ============================================================
# Validate method (basic checks that should already pass)
# ============================================================


def test_validate_empty_shortcut():
    s = Shortcut("Test")
    warnings = s.validate()
    assert any("no actions" in w.lower() for w in warnings)


def test_validate_clean_shortcut():
    s = Shortcut("Test")
    s.add(actions.make("comment", WFCommentActionText="Hello"))
    warnings = s.validate()
    assert warnings == [], f"Expected no warnings, got: {warnings}"


def test_validate_ordering():
    """Check 2: Mode 2 before Mode 0 for same group should warn."""
    s = Shortcut("Test")
    gid = "TEST-ORDER-GROUP"
    s.actions.append(
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": gid,
                "WFControlFlowMode": 2,
            },
        }
    )
    s.actions.append(
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": gid,
                "WFControlFlowMode": 0,
                "WFCondition": 100,
            },
        }
    )
    warnings = s.validate()
    assert any("before start" in w for w in warnings), (
        f"Expected ordering warning, got: {warnings}"
    )


def test_validate_interleaving():
    """Check 5: Interleaved blocks should warn."""
    s = Shortcut("Test")
    gid_a = "GROUP-A-INTERLEAVE"
    gid_b = "GROUP-B-INTERLEAVE"
    # A-start, B-start, A-end, B-end = interleaved
    s.actions.extend(
        [
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid_a,
                    "WFControlFlowMode": 0,
                    "WFCondition": 100,
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid_b,
                    "WFControlFlowMode": 0,
                    "WFCondition": 100,
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid_a,
                    "WFControlFlowMode": 2,
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid_b,
                    "WFControlFlowMode": 2,
                },
            },
        ]
    )
    warnings = s.validate()
    assert any(
        "interleaved" in w.lower() or "overlap" in w.lower() for w in warnings
    ), f"Expected interleaving warning, got: {warnings}"


def test_validate_mode1_outside_range():
    """Check 3: Mode 1 marker outside its group's range should warn."""
    s = Shortcut("Test")
    gid = "TEST-MODE1-RANGE"
    # Mode 1 before Mode 0 — should warn
    s.actions.extend(
        [
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 1,
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 0,
                    "WFCondition": 100,
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.comment",
                "WFWorkflowActionParameters": {"WFCommentActionText": "body"},
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 2,
                },
            },
        ]
    )
    warnings = s.validate()
    assert any("outside" in w.lower() for w in warnings), (
        f"Expected out-of-range Mode 1 warning, got: {warnings}"
    )


def test_validate_menu_case_mismatch():
    """Check 4: Menu with wrong number of case markers should warn."""
    s = Shortcut("Test")
    gid = "TEST-MENU-MISMATCH"
    # Menu declares 3 items but only has 2 case markers
    s.actions.extend(
        [
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 0,
                    "WFMenuPrompt": "Pick:",
                    "WFMenuItems": ["A", "B", "C"],
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 1,
                    "WFMenuItemTitle": "A",
                },
            },
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 1,
                    "WFMenuItemTitle": "B",
                },
            },
            # Missing case C
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": gid,
                    "WFControlFlowMode": 2,
                },
            },
        ]
    )
    warnings = s.validate()
    assert any("3 items but 2 case" in w for w in warnings), (
        f"Expected menu case mismatch warning, got: {warnings}"
    )


# ============================================================
# Condition mapping
# ============================================================


def test_condition_map_coverage():
    """All documented conditions should map to integers."""
    for name in [
        "has_any_value",
        "does_not_have_any_value",
        "equals_number",
        "is_greater_than",
        "is_less_than",
        "equals_string",
        "not_equal_string",
        "contains",
        "does_not_contain",
        "is_before",
        "==",
        "!=",
        ">",
        "<",
    ]:
        assert name in CONDITION_MAP, f"Missing condition: {name}"
        assert isinstance(CONDITION_MAP[name], int), (
            f"Condition {name} should map to int"
        )


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("Apple Shortcuts Compiler — Unit Tests")
    print("=" * 50)

    tests = [
        # Phase 1A: bool/int fix
        ("bool_before_int", test_bool_before_int),
        ("dict_and_list_types", test_dict_and_list_types),
        # Action resolution
        ("resolve_short_name", test_resolve_short_name),
        ("resolve_dotted_name", test_resolve_dotted_name),
        ("resolve_full_identifier", test_resolve_full_identifier),
        ("resolve_third_party", test_resolve_third_party),
        ("resolve_unknown_raises", test_resolve_unknown_raises),
        # Parameter validation
        ("make_rejects_unknown_params", test_make_rejects_unknown_params),
        ("make_accepts_valid_params", test_make_accepts_valid_params),
        # Control flow
        ("if_block_structure", test_if_block_structure),
        ("if_else_block_structure", test_if_else_block_structure),
        ("menu_block_structure", test_menu_block_structure),
        ("nested_control_flow", test_nested_control_flow),
        # Builders
        ("build_list", test_build_list),
        ("build_headers", test_build_headers),
        ("build_quantity", test_build_quantity),
        ("build_token_string", test_build_token_string),
        # Validate
        ("validate_empty_shortcut", test_validate_empty_shortcut),
        ("validate_clean_shortcut", test_validate_clean_shortcut),
        ("validate_ordering", test_validate_ordering),
        ("validate_interleaving", test_validate_interleaving),
        ("validate_mode1_outside_range", test_validate_mode1_outside_range),
        ("validate_menu_case_mismatch", test_validate_menu_case_mismatch),
        # Conditions
        ("condition_map_coverage", test_condition_map_coverage),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")
