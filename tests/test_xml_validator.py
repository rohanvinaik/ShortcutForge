"""
XML Shortcuts Validator Tests
==============================
Fixture-based deterministic tests for Shortcut.validate().

Tests cover:
  - Clean shortcuts produce zero warnings
  - Unclosed control flow blocks detected
  - Orphan end blocks detected
  - End-before-start ordering detected
  - Menu case count mismatches detected
  - UUID collision handling
  - Empty shortcut detection

Optional stress test mode (requires XML files):
  python3 scripts/test_xml_validator.py --stress --xml-dir /path/to/xmls

Run: python3 scripts/test_xml_validator.py
"""
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from shortcuts_compiler import Shortcut


# -- Test Harness ----------------------------------------------------------

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


# -- Fixture Helpers -------------------------------------------------------

def _make_action(identifier: str, **params) -> dict:
    """Create a minimal action dict."""
    action = {"WFWorkflowActionIdentifier": identifier}
    if params:
        action["WFWorkflowActionParameters"] = params
    return action


def _make_cf_action(identifier: str, gid: str, mode: int, **extra_params) -> dict:
    """Create a control flow action with GroupingIdentifier and WFControlFlowMode."""
    params = {"GroupingIdentifier": gid, "WFControlFlowMode": mode}
    params.update(extra_params)
    return _make_action(identifier, **params)


def _make_shortcut_with_actions(name: str, actions_list: list[dict]) -> Shortcut:
    """Create a Shortcut with injected actions for validation testing."""
    s = Shortcut(name)
    s.actions = actions_list
    s.envelope["WFWorkflowActions"] = actions_list
    s._action_stack = [actions_list]
    return s


# ==========================================================================
# Clean Shortcut Tests
# ==========================================================================

def test_clean_simple_shortcut():
    """Simple shortcut with no control flow validates cleanly."""
    actions = [
        _make_action("is.workflow.actions.comment", WFCommentActionText="Hello"),
        _make_action("is.workflow.actions.notification",
                     WFNotificationActionTitle="Done"),
    ]
    s = _make_shortcut_with_actions("Clean Simple", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("clean_simple_shortcut", test_clean_simple_shortcut)


def test_clean_if_block():
    """Properly paired IF/ELSE/ENDIF validates cleanly."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid, 0),  # IF start
        _make_action("is.workflow.actions.comment", WFCommentActionText="then"),
        _make_cf_action("is.workflow.actions.conditional", gid, 1),  # ELSE
        _make_action("is.workflow.actions.comment", WFCommentActionText="else"),
        _make_cf_action("is.workflow.actions.conditional", gid, 2),  # ENDIF
    ]
    s = _make_shortcut_with_actions("Clean IF", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("clean_if_block", test_clean_if_block)


def test_clean_nested_blocks():
    """Properly nested IF blocks validate cleanly."""
    gid1 = str(uuid.uuid4())
    gid2 = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid1, 0),  # outer IF
        _make_cf_action("is.workflow.actions.conditional", gid2, 0),  # inner IF
        _make_cf_action("is.workflow.actions.conditional", gid2, 2),  # inner ENDIF
        _make_cf_action("is.workflow.actions.conditional", gid1, 2),  # outer ENDIF
    ]
    s = _make_shortcut_with_actions("Nested", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("clean_nested_blocks", test_clean_nested_blocks)


def test_clean_menu_block():
    """Properly formed menu with matching case count validates cleanly."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 0,
                        WFMenuItems=["Option A", "Option B"]),
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 1),  # Case A
        _make_action("is.workflow.actions.comment", WFCommentActionText="A"),
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 1),  # Case B
        _make_action("is.workflow.actions.comment", WFCommentActionText="B"),
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 2),  # ENDMENU
    ]
    s = _make_shortcut_with_actions("Clean Menu", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("clean_menu_block", test_clean_menu_block)


def test_clean_repeat_block():
    """Properly paired REPEAT/ENDREPEAT validates cleanly."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.repeat.count", gid, 0),  # REPEAT start
        _make_action("is.workflow.actions.comment", WFCommentActionText="loop body"),
        _make_cf_action("is.workflow.actions.repeat.count", gid, 2),  # ENDREPEAT
    ]
    s = _make_shortcut_with_actions("Clean Repeat", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("clean_repeat_block", test_clean_repeat_block)


# ==========================================================================
# Unclosed Block Tests
# ==========================================================================

def test_unclosed_if_block():
    """IF without ENDIF detected."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid, 0),  # IF start
        _make_action("is.workflow.actions.comment", WFCommentActionText="oops"),
        # Missing ENDIF (mode 2)
    ]
    s = _make_shortcut_with_actions("Unclosed IF", actions)
    warnings = s.validate()
    assert any("unclosed" in w.lower() or "Unclosed" in w for w in warnings), \
        f"Expected unclosed warning, got: {warnings}"

run_test("unclosed_if_block", test_unclosed_if_block)


def test_orphan_end_block():
    """ENDIF without matching IF detected."""
    gid = str(uuid.uuid4())
    actions = [
        _make_action("is.workflow.actions.comment", WFCommentActionText="before"),
        _make_cf_action("is.workflow.actions.conditional", gid, 2),  # orphan ENDIF
    ]
    s = _make_shortcut_with_actions("Orphan End", actions)
    warnings = s.validate()
    assert any("without" in w.lower() or "End blocks" in w for w in warnings), \
        f"Expected orphan end warning, got: {warnings}"

run_test("orphan_end_block", test_orphan_end_block)


# ==========================================================================
# Ordering Tests
# ==========================================================================

def test_end_before_start():
    """End block before start block detected."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid, 2),  # ENDIF first
        _make_cf_action("is.workflow.actions.conditional", gid, 0),  # IF after
    ]
    s = _make_shortcut_with_actions("End Before Start", actions)
    warnings = s.validate()
    assert any("before start" in w.lower() or "before" in w.lower() for w in warnings), \
        f"Expected ordering warning, got: {warnings}"

run_test("end_before_start", test_end_before_start)


# ==========================================================================
# Menu Validation Tests
# ==========================================================================

def test_menu_case_mismatch():
    """Menu with wrong number of cases detected."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 0,
                        WFMenuItems=["A", "B", "C"]),  # 3 items
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 1),  # Case 1
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 1),  # Case 2
        # Missing case 3
        _make_cf_action("is.workflow.actions.choosefrommenu", gid, 2),  # ENDMENU
    ]
    s = _make_shortcut_with_actions("Menu Mismatch", actions)
    warnings = s.validate()
    has_mismatch_warning = any(
        "menu" in w.lower() and ("mismatch" in w.lower() or "case" in w.lower() or "items" in w.lower())
        for w in warnings
    )
    assert has_mismatch_warning, f"Expected menu case mismatch warning, got: {warnings}"

run_test("menu_case_mismatch", test_menu_case_mismatch)


# ==========================================================================
# Empty Shortcut Test
# ==========================================================================

def test_empty_shortcut():
    """Empty shortcut produces a warning."""
    s = _make_shortcut_with_actions("Empty", [])
    warnings = s.validate()
    assert any("no actions" in w.lower() for w in warnings), \
        f"Expected 'no actions' warning, got: {warnings}"

run_test("empty_shortcut", test_empty_shortcut)


# ==========================================================================
# UUID Collision / Edge Cases
# ==========================================================================

def test_uuid_collision_no_crash():
    """Duplicate GroupingIdentifier for unrelated blocks doesn't crash."""
    gid = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid, 0),
        _make_cf_action("is.workflow.actions.conditional", gid, 2),
        _make_cf_action("is.workflow.actions.conditional", gid, 0),
        _make_cf_action("is.workflow.actions.conditional", gid, 2),
    ]
    s = _make_shortcut_with_actions("UUID Collision", actions)
    warnings = s.validate()
    assert isinstance(warnings, list)

run_test("uuid_collision_no_crash", test_uuid_collision_no_crash)


def test_multiple_clean_blocks():
    """Multiple independent clean control flow blocks validate cleanly."""
    gid1 = str(uuid.uuid4())
    gid2 = str(uuid.uuid4())
    actions = [
        _make_cf_action("is.workflow.actions.conditional", gid1, 0),
        _make_cf_action("is.workflow.actions.conditional", gid1, 2),
        _make_cf_action("is.workflow.actions.repeat.count", gid2, 0),
        _make_cf_action("is.workflow.actions.repeat.count", gid2, 2),
    ]
    s = _make_shortcut_with_actions("Multiple Clean", actions)
    warnings = s.validate()
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {warnings}"

run_test("multiple_clean_blocks", test_multiple_clean_blocks)


# ==========================================================================
# Stress Test Mode (optional, requires XML files)
# ==========================================================================

def _run_stress_test(xml_dir: str):
    """Run the original stress test against real XML shortcuts."""
    import glob
    import plistlib

    xml_dir = os.path.abspath(xml_dir)
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    if not xml_files:
        print(f"\n  No XML files found in {xml_dir}, skipping stress test")
        return

    print(f"\n  Stress test: {len(xml_files)} XML files in {xml_dir}")
    processed = 0
    clean = 0
    warnings_by_file = {}

    for fpath in xml_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "rb") as f:
                plist = plistlib.load(f)
        except Exception:
            continue

        actions_list = plist.get("WFWorkflowActions", [])
        if not actions_list:
            continue

        processed += 1
        s = _make_shortcut_with_actions(fname.replace(".xml", ""), actions_list)
        warnings = s.validate()
        if warnings:
            warnings_by_file[fname] = warnings
        else:
            clean += 1

    print(f"  Processed: {processed}, Clean: {clean}, With warnings: {len(warnings_by_file)}")
    if warnings_by_file:
        for fname, ws in sorted(warnings_by_file.items()):
            print(f"    {fname}: {len(ws)} warning(s)")
            for w in ws[:3]:
                print(f"      - {w}")


# -- Report ----------------------------------------------------------------

print()
print("=" * 50)
print(f"Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
if _fail > 0:
    print(f"FAILURES: {_fail}")
    for name, passed, err in _results:
        if not passed:
            print(f"  {name}: {err}")
    sys.exit(1)
else:
    print("All tests passed.")

# Optional: run stress test if --stress flag provided
if "--stress" in sys.argv:
    xml_dir = os.path.expanduser("~/Downloads")
    if "--xml-dir" in sys.argv:
        dir_idx = sys.argv.index("--xml-dir")
        if dir_idx + 1 < len(sys.argv):
            xml_dir = sys.argv[dir_idx + 1]
    _run_stress_test(xml_dir)
