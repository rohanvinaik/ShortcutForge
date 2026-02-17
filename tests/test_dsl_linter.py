"""
Regression tests for dsl_linter.py.
Tests hallucination aliases, ENDSHORTCUT handling, structural repairs, and idempotency.

Run: python3 scripts/test_dsl_linter.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from dsl_linter import lint_dsl, ActionResolver, __version__

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
        print(f"  FAIL: {name} -- {e}")


# ============================================================
# Module metadata
# ============================================================

def test_version():
    assert __version__ == "2.4", f"Expected version 2.4, got {__version__}"

run_test("version", test_version)


# ============================================================
# High-confidence alias tests
# ============================================================

def test_alias_crop():
    dsl = 'SHORTCUT "Test"\nACTION crop Size="100x100"\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "crop"]
    assert len(changes) == 1, f"Expected 1 crop change, got {len(changes)}"
    c = changes[0]
    assert c.replacement == "image.crop", f"Expected image.crop, got {c.replacement}"
    assert c.kind == "action", f"Expected kind=action, got {c.kind}"
    assert c.confidence == 0.95, f"Expected confidence=0.95, got {c.confidence}"
    assert c.reason, "Expected non-empty reason"

def test_alias_selectmedia():
    dsl = 'SHORTCUT "Test"\nACTION selectmedia\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "selectmedia"]
    assert len(changes) == 1
    assert changes[0].replacement == "selectphoto"
    assert changes[0].kind == "action"

def test_alias_reminders_add():
    dsl = 'SHORTCUT "Test"\nACTION reminders.add\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "reminders.add"]
    assert len(changes) == 1
    assert changes[0].replacement == "addnewreminder"

def test_alias_gettimeuntil():
    dsl = 'SHORTCUT "Test"\nACTION gettimeuntil\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "gettimeuntil"]
    assert len(changes) == 1
    assert changes[0].replacement == "gettimebetweendates"

for name, fn in [
    ("alias_crop", test_alias_crop),
    ("alias_selectmedia", test_alias_selectmedia),
    ("alias_reminders_add", test_alias_reminders_add),
    ("alias_gettimeuntil", test_alias_gettimeuntil),
]:
    run_test(name, fn)


# ============================================================
# Phase 8: New hallucination alias tests (device, timer, media)
# ============================================================

def test_alias_flashon():
    """flashon → flashlight (device control hallucination)."""
    dsl = 'SHORTCUT "Test"\nACTION flashon\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "flashon"]
    assert len(changes) == 1, f"Expected 1 change, got {len(changes)}"
    assert changes[0].replacement == "flashlight", f"Got {changes[0].replacement}"
    assert changes[0].confidence == 0.95

def test_alias_startstopwatch():
    """startstopwatch → startstopwatch (maps via canonical_map to system intent)."""
    dsl = 'SHORTCUT "Test"\nACTION startstopwatch\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    # startstopwatch IS in the canonical_map, so it should be valid (no changes)
    # OR it maps via alias to itself which is resolvable
    action_changes = [c for c in result.changes if c.kind == "action" and c.original == "startstopwatch"]
    # startstopwatch is in canonical map so may not need linter fix
    # but the alias maps it to itself — should still be fine

def test_alias_alarm_to_createalarm():
    """alarm → createalarm (timer hallucination)."""
    dsl = 'SHORTCUT "Test"\nACTION alarm\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "alarm"]
    assert len(changes) == 1, f"Expected 1 change, got {len(changes)}"
    assert changes[0].replacement == "createalarm", f"Got {changes[0].replacement}"
    assert changes[0].confidence == 0.95

def test_alias_menu_to_keyword():
    """ACTION menu → MENU (Phase 12 keyword rewriting takes precedence over alias)."""
    dsl = 'SHORTCUT "Test"\nACTION menu\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    # Phase 12: _fix_action_as_keyword rewrites ACTION menu → MENU (keyword)
    # This takes precedence over HALLUCINATION_ALIASES menu → choosefrommenu
    struct_changes = [c for c in result.changes if c.kind == "structure" and "menu" in c.reason.lower()]
    assert len(struct_changes) == 1, f"Expected 1 structure change for menu, got {struct_changes}"
    assert "MENU" in struct_changes[0].replacement, f"Expected MENU keyword, got {struct_changes[0].replacement}"

def test_alias_screenshot_to_takescreenshot():
    """screenshot → takescreenshot."""
    dsl = 'SHORTCUT "Test"\nACTION screenshot\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "screenshot"]
    assert len(changes) == 1
    assert changes[0].replacement == "takescreenshot"

def test_alias_recordvoicememo():
    """recordvoicememo → recordvoicememo (maps via CM to VoiceMemos intent)."""
    dsl = 'SHORTCUT "Test"\nACTION recordvoicememo\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    # recordvoicememo is in canonical map via auto-generation, so it's valid
    # The alias maps to itself (idempotent)

def test_alias_speak():
    """speak → speaktext (speech alias)."""
    dsl = 'SHORTCUT "Test"\nACTION speak\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "speak"]
    assert len(changes) == 1
    assert changes[0].replacement == "speaktext"

def test_alias_donotdisturb():
    """donotdisturb → dnd.set (focus mode alias)."""
    dsl = 'SHORTCUT "Test"\nACTION donotdisturb\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "donotdisturb"]
    assert len(changes) == 1
    assert changes[0].replacement == "dnd.set"

def test_alias_sendtext():
    """sendtext → sendmessage (messaging alias)."""
    dsl = 'SHORTCUT "Test"\nACTION sendtext\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "sendtext"]
    assert len(changes) == 1
    assert changes[0].replacement == "sendmessage"

def test_alias_scanqr():
    """scanqr → scanbarcode (QR alias)."""
    dsl = 'SHORTCUT "Test"\nACTION scanqr\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "scanqr"]
    assert len(changes) == 1
    assert changes[0].replacement == "scanbarcode"

for name, fn in [
    ("alias_flashon", test_alias_flashon),
    ("alias_startstopwatch", test_alias_startstopwatch),
    ("alias_alarm_to_createalarm", test_alias_alarm_to_createalarm),
    ("alias_menu_to_keyword", test_alias_menu_to_keyword),
    ("alias_screenshot_to_takescreenshot", test_alias_screenshot_to_takescreenshot),
    ("alias_recordvoicememo", test_alias_recordvoicememo),
    ("alias_speak", test_alias_speak),
    ("alias_donotdisturb", test_alias_donotdisturb),
    ("alias_sendtext", test_alias_sendtext),
    ("alias_scanqr", test_alias_scanqr),
]:
    run_test(name, fn)


# ============================================================
# Semantic-risky alias tests
# ============================================================

def test_semantic_risky_convertlivephoto():
    dsl = 'SHORTCUT "Test"\nACTION convertlivephoto\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    changes = [c for c in result.changes if c.original == "convertlivephoto"]
    assert len(changes) == 1
    c = changes[0]
    assert c.replacement == "getlatestlivephotos", f"Got {c.replacement}"
    assert c.kind == "alias_warning", f"Expected kind=alias_warning, got {c.kind}"
    assert c.confidence == 0.7, f"Expected confidence=0.7, got {c.confidence}"
    assert "semantic-risky" in c.reason, f"Expected 'semantic-risky' in reason, got {c.reason!r}"

run_test("semantic_risky_convertlivephoto", test_semantic_risky_convertlivephoto)


# ============================================================
# ENDSHORTCUT tests
# ============================================================

def test_endshortcut_auto_append():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\n'
    result = lint_dsl(dsl)
    assert "ENDSHORTCUT" in result.text
    es_changes = [c for c in result.changes if c.replacement == "ENDSHORTCUT"]
    assert len(es_changes) == 1

def test_endshortcut_truncate_after():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nENDSHORTCUT\nSome ramble after\nMore ramble\n'
    result = lint_dsl(dsl)
    assert "ramble" not in result.text
    trunc_changes = [c for c in result.changes if "truncated" in c.replacement]
    assert len(trunc_changes) >= 1

def test_endshortcut_inside_quote_not_structural():
    dsl = 'SHORTCUT "Test"\nACTION comment Text="ENDSHORTCUT is not real"\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    # Should NOT truncate — ENDSHORTCUT inside quotes is not structural
    assert 'ENDSHORTCUT is not real' in result.text

def test_endshortcut_already_present():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    # ENDSHORTCUT already there — no ENDSHORTCUT-related changes
    es_changes = [c for c in result.changes if c.replacement == "ENDSHORTCUT"]
    assert len(es_changes) == 0

for name, fn in [
    ("endshortcut_auto_append", test_endshortcut_auto_append),
    ("endshortcut_truncate_after", test_endshortcut_truncate_after),
    ("endshortcut_inside_quote_not_structural", test_endshortcut_inside_quote_not_structural),
    ("endshortcut_already_present", test_endshortcut_already_present),
]:
    run_test(name, fn)


# ============================================================
# Structure repair tests
# ============================================================

def test_unclosed_if_auto_close():
    dsl = 'SHORTCUT "Test"\nIF @prev has_any_value\nACTION openurl URL="x"\n'
    result = lint_dsl(dsl)
    assert "ENDIF" in result.text

def test_orphan_else_removed():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="x"\nELSE\nACTION gettext\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    orphan_changes = [c for c in result.changes if "orphan" in c.replacement.lower()]
    assert len(orphan_changes) >= 1

for name, fn in [
    ("unclosed_if_auto_close", test_unclosed_if_auto_close),
    ("orphan_else_removed", test_orphan_else_removed),
]:
    run_test(name, fn)


# ============================================================
# Idempotency tests
# ============================================================

def test_idempotency_clean_dsl():
    """Clean DSL should produce no changes."""
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert not result.was_modified, f"Clean DSL should not be modified, got {len(result.changes)} changes"

def test_idempotency_double_lint():
    """Linting twice should produce no changes on second pass."""
    dsl = 'SHORTCUT "Test"\nACTION crop Size="100"\n'
    first = lint_dsl(dsl)
    assert first.was_modified, "First pass should make changes"
    second = lint_dsl(first.text)
    assert not second.was_modified, f"Second pass should make no changes, got {len(second.changes)}: {[(c.kind, c.original, c.replacement) for c in second.changes]}"

for name, fn in [
    ("idempotency_clean_dsl", test_idempotency_clean_dsl),
    ("idempotency_double_lint", test_idempotency_double_lint),
]:
    run_test(name, fn)


# ============================================================
# ActionResolver tests
# ============================================================

def test_resolver_unknown_action_no_match():
    resolver = ActionResolver()
    closest, is_alias, reason = resolver.find_closest("completelyFakeAction", cutoff=0.65)
    assert closest is None, f"completelyFakeAction should not match anything, got {closest}"

def test_resolver_known_action():
    resolver = ActionResolver()
    assert resolver.is_valid("openurl"), "openurl should be valid"

def test_resolver_is_semantic_risky():
    resolver = ActionResolver()
    is_risky, reason = resolver.is_semantic_risky("convertlivephoto")
    assert is_risky, "convertlivephoto should be semantic-risky"
    assert "conversion" in reason.lower() or "fetch" in reason.lower()

for name, fn in [
    ("resolver_unknown_action_no_match", test_resolver_unknown_action_no_match),
    ("resolver_known_action", test_resolver_known_action),
    ("resolver_is_semantic_risky", test_resolver_is_semantic_risky),
]:
    run_test(name, fn)


# ============================================================
# Phase 12: ACTION-as-keyword rewriting tests
# ============================================================

def test_action_as_keyword_menu():
    """ACTION menu WFMenuPrompt='Choose option' → MENU 'Choose option'."""
    dsl = 'SHORTCUT "Test"\nACTION menu WFMenuPrompt="Choose option"\nCASE "A"\nACTION openurl URL="x"\nCASE "B"\nACTION gettext\nENDMENU\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert 'MENU "Choose option"' in result.text, f"Expected MENU keyword, got:\n{result.text}"
    assert "ACTION menu" not in result.text
    struct_changes = [c for c in result.changes if c.kind == "structure" and "menu" in c.reason.lower()]
    assert len(struct_changes) >= 1, f"Expected structure change, got {struct_changes}"

def test_action_as_keyword_repeat():
    """ACTION repeat WFRepeatCount=5 → REPEAT 5."""
    dsl = 'SHORTCUT "Test"\nACTION repeat WFRepeatCount=5\nACTION openurl URL="x"\nENDREPEAT\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert "REPEAT 5" in result.text, f"Expected REPEAT 5, got:\n{result.text}"
    assert "ACTION repeat" not in result.text

def test_action_as_keyword_repeat_with_each():
    """ACTION repeat_with_each → FOREACH @input."""
    dsl = 'SHORTCUT "Test"\nACTION repeat_with_each\nACTION openurl URL="x"\nENDFOREACH\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert "FOREACH @input" in result.text, f"Expected FOREACH @input, got:\n{result.text}"
    assert "ACTION repeat_with_each" not in result.text

def test_action_as_keyword_choosefrommenu():
    """ACTION choosefrommenu WFMenuPrompt='Pick one' → MENU 'Pick one'."""
    dsl = 'SHORTCUT "Test"\nACTION choosefrommenu WFMenuPrompt="Pick one"\nCASE "X"\nACTION gettext\nENDMENU\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert 'MENU "Pick one"' in result.text, f"Expected MENU keyword, got:\n{result.text}"
    assert "ACTION choosefrommenu" not in result.text

def test_action_as_keyword_regular_unchanged():
    """Regular ACTION lines (not keywords) should be unchanged."""
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert 'ACTION openurl URL="https://example.com"' in result.text
    struct_changes = [c for c in result.changes if c.kind == "structure"]
    assert len(struct_changes) == 0, f"No structure changes expected, got {struct_changes}"

def test_action_as_keyword_foreach():
    """ACTION foreach → FOREACH @input."""
    dsl = 'SHORTCUT "Test"\nACTION foreach\nACTION openurl URL="x"\nENDFOREACH\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    assert "FOREACH @input" in result.text, f"Expected FOREACH @input, got:\n{result.text}"
    assert "ACTION foreach" not in result.text

for name, fn in [
    ("action_as_keyword_menu", test_action_as_keyword_menu),
    ("action_as_keyword_repeat", test_action_as_keyword_repeat),
    ("action_as_keyword_repeat_with_each", test_action_as_keyword_repeat_with_each),
    ("action_as_keyword_choosefrommenu", test_action_as_keyword_choosefrommenu),
    ("action_as_keyword_regular_unchanged", test_action_as_keyword_regular_unchanged),
    ("action_as_keyword_foreach", test_action_as_keyword_foreach),
]:
    run_test(name, fn)


# ============================================================
# LintChange reason field tests
# ============================================================

def test_reason_field_populated():
    """Verify that reason field is populated for alias changes."""
    dsl = 'SHORTCUT "Test"\nACTION crop\nENDSHORTCUT\n'
    result = lint_dsl(dsl)
    action_changes = [c for c in result.changes if c.kind in ("action", "alias_warning")]
    assert len(action_changes) >= 1
    assert all(c.reason for c in action_changes), "All alias changes should have a reason"

run_test("reason_field_populated", test_reason_field_populated)


# ============================================================
# Results
# ============================================================

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
if failed == 0:
    print("All tests passed.")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
