"""
Round-Trip Reconstruction Tests
================================
Pick real shortcuts of varying complexity, reconstruct them using the
compiler API, then verify:
  1. Action identifier sequence matches exactly
  2. Parameter keys match (excluding compiler-injected keys)
  3. Scalar parameter values match where feasible
  4. Control flow grouping structure matches

Shortcuts tested:
  - Find_gas_nearby.shortcut         (5 actions, linear pipeline)
  - Log_toothbrushing.shortcut       (9 actions, menu with output wiring)
  - Check_upcoming_Golden_Hour_times (15 actions, nested repeat-each)

Run: python3 scripts/test_roundtrip.py
"""

import os
import plistlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from shortcuts_compiler import (
    ActionHandle,
    Shortcut,
    actions,
    ref_extension_input,
    ref_variable,
    wrap_token_attachment,
)

DOWNLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "downloaded"
)

# Keys injected by the compiler or inherently non-deterministic
SKIP_KEYS = {"UUID", "GroupingIdentifier"}

passed = 0
failed = 0


def load_original(filename):
    """Load a downloaded shortcut and return its action list."""
    path = os.path.join(DOWNLOAD_DIR, filename)
    with open(path, "rb") as f:
        plist = plistlib.load(f)
    return plist.get("WFWorkflowActions", [])


def compare_actions(original, reconstructed):
    """
    Compare two action lists structurally.
    Returns (passed_checks, failed_checks, details).
    """
    checks_passed = 0
    checks_failed = 0
    details = []

    # Check 1: Same number of actions
    if len(original) != len(reconstructed):
        details.append(
            f"Action count mismatch: original={len(original)}, reconstructed={len(reconstructed)}"
        )
        checks_failed += 1
    else:
        checks_passed += 1

    # Check 2: Action identifier sequence matches
    orig_idents = [a.get("WFWorkflowActionIdentifier", "") for a in original]
    recon_idents = [a.get("WFWorkflowActionIdentifier", "") for a in reconstructed]

    if orig_idents == recon_idents:
        checks_passed += 1
    else:
        checks_failed += 1
        for i, (o, r) in enumerate(zip(orig_idents, recon_idents)):
            if o != r:
                details.append(
                    f"Identifier mismatch at [{i}]: original={o}, reconstructed={r}"
                )
                break
        if len(orig_idents) != len(recon_idents):
            details.append(
                f"Identifier count: original={len(orig_idents)}, reconstructed={len(recon_idents)}"
            )

    # Check 3: Parameter keys match for each action (excluding SKIP_KEYS)
    min_len = min(len(original), len(reconstructed))
    param_mismatches = 0
    for i in range(min_len):
        orig_params = (
            set(original[i].get("WFWorkflowActionParameters", {}).keys()) - SKIP_KEYS
        )
        recon_params = (
            set(reconstructed[i].get("WFWorkflowActionParameters", {}).keys())
            - SKIP_KEYS
        )
        if orig_params != recon_params:
            missing = orig_params - recon_params
            extra = recon_params - orig_params
            short_ident = (
                orig_idents[i].replace("is.workflow.actions.", "")
                if i < len(orig_idents)
                else "?"
            )
            if missing:
                details.append(f"[{i}] {short_ident}: missing params {missing}")
            if extra:
                details.append(f"[{i}] {short_ident}: extra params {extra}")
            param_mismatches += 1

    if param_mismatches == 0:
        checks_passed += 1
    else:
        checks_failed += 1
        details.append(
            f"Parameter key mismatches in {param_mismatches}/{min_len} actions"
        )

    # Check 4: Scalar parameter values match where applicable
    scalar_total = 0
    scalar_match = 0
    for i in range(min_len):
        orig_p = original[i].get("WFWorkflowActionParameters", {})
        recon_p = reconstructed[i].get("WFWorkflowActionParameters", {})
        for key in set(orig_p.keys()) & set(recon_p.keys()) - SKIP_KEYS:
            orig_val = orig_p[key]
            recon_val = recon_p[key]
            # Only compare scalars (str, int, float, bool, list of strings)
            if isinstance(orig_val, (str, int, float, bool)):
                scalar_total += 1
                if orig_val == recon_val:
                    scalar_match += 1
                else:
                    short_ident = (
                        orig_idents[i].replace("is.workflow.actions.", "")
                        if i < len(orig_idents)
                        else "?"
                    )
                    details.append(
                        f"[{i}] {short_ident}.{key}: {repr(orig_val)[:60]} != {repr(recon_val)[:60]}"
                    )
            elif isinstance(orig_val, list) and all(
                isinstance(x, str) for x in orig_val
            ):
                scalar_total += 1
                if orig_val == recon_val:
                    scalar_match += 1
                else:
                    short_ident = (
                        orig_idents[i].replace("is.workflow.actions.", "")
                        if i < len(orig_idents)
                        else "?"
                    )
                    details.append(f"[{i}] {short_ident}.{key}: list mismatch")

    if scalar_total > 0:
        if scalar_match == scalar_total:
            checks_passed += 1
        else:
            checks_failed += 1
            details.append(f"Scalar value matches: {scalar_match}/{scalar_total}")
    else:
        checks_passed += 1  # No scalars to compare

    # Check 5: Control flow structure — GroupingIdentifier pairing
    def get_group_structure(action_list):
        groups = {}
        for a in action_list:
            p = a.get("WFWorkflowActionParameters", {})
            gid = p.get("GroupingIdentifier")
            mode = p.get("WFControlFlowMode")
            if gid is not None and mode is not None:
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(mode)
        return sorted([sorted(modes) for modes in groups.values()])

    orig_groups = get_group_structure(original)
    recon_groups = get_group_structure(reconstructed)

    if orig_groups == recon_groups:
        checks_passed += 1
    else:
        checks_failed += 1
        details.append(
            f"Control flow groups differ: original={orig_groups}, reconstructed={recon_groups}"
        )

    return checks_passed, checks_failed, details


# ============================================================
# Test 1: Find Gas Nearby (5 actions, linear)
# ============================================================


def reconstruct_find_gas():
    """Reconstruct Find_gas_nearby.shortcut.

    Tests: linear pipeline, ActionHandle auto-wiring, extension input.
    """
    # Load original to get exact comment text (preserves Unicode)
    orig = load_original("Find_gas_nearby.shortcut")

    s = Shortcut("Find Gas Nearby")

    # [0] Comment
    s.add(
        actions.make(
            "comment",
            WFCommentActionText=orig[0]["WFWorkflowActionParameters"][
                "WFCommentActionText"
            ],
        )
    )

    # [1] Comment (long, with smart quotes — copy from original)
    s.add(
        actions.make(
            "comment",
            WFCommentActionText=orig[1]["WFWorkflowActionParameters"][
                "WFCommentActionText"
            ],
        )
    )

    # [2] Search local businesses using extension input as location
    businesses = s.add(
        actions.make(
            "searchlocalbusinesses",
            WFSearchQuery="Gas Station",
            WFInput=ref_extension_input(),
        )
    )

    # [3] Choose from the results
    chosen = s.add(
        actions.make(
            "choosefromlist",
            WFChooseFromListActionPrompt="Which gas station do you want directions for?",
            WFInput=businesses,
        )
    )

    # [4] Get directions
    s.add(
        actions.make(
            "getdirections", WFGetDirectionsActionMode="Driving", WFDestination=chosen
        )
    )

    return s


# ============================================================
# Test 2: Log Toothbrushing (9 actions, menu)
# ============================================================


def reconstruct_log_toothbrushing():
    """Reconstruct Log_toothbrushing.shortcut.

    Tests: menu_block context manager, menu output reference, quantity builder.
    """
    s = Shortcut("Log Toothbrushing")

    # [0] Comment
    s.add(
        actions.make(
            "comment",
            WFCommentActionText="Adds data to Health that you brushed you teeth, either at the current time or an earlier date you input.",
        )
    )

    # [1-6] Menu: "Now" or "Earlier"
    with s.menu_block("When did you brush?", ["Now", "Earlier"]) as cases:
        # Case: Now [2]
        cases["Now"]()
        # [3] Current Date
        s.add(actions.make("date"))

        # Case: Earlier [4]
        cases["Earlier"]()
        # [5] Ask for time
        s.add(actions.make("ask", WFAskActionPrompt="When?", WFInputType="Time"))

    # After the menu block, the menu end action is s.actions[-1].
    # Its UUID is the output of the choosefrommenu block ("Menu Result").
    menu_end_uuid = s.actions[-1]["WFWorkflowActionParameters"].get("UUID")
    if not menu_end_uuid:
        # The menu end action's UUID is set on the action itself
        # For choosefrommenu actions injected by menu_block, UUID may not be set.
        # But the action was appended without going through s.add(), so no UUID.
        # We need to add one manually.
        import uuid as uuid_mod

        menu_end_uuid = str(uuid_mod.uuid4()).upper()
        s.actions[-1]["WFWorkflowActionParameters"]["UUID"] = menu_end_uuid

    menu_result = ActionHandle(
        menu_end_uuid, "Menu Result", "is.workflow.actions.choosefrommenu"
    )

    # [7] Adjust date: subtract 2 minutes from the chosen time
    adjusted = s.add(
        actions.make(
            "adjustdate",
            WFDate=menu_result,
            WFDuration=actions.build_quantity(2, "min"),
        )
    )

    # [8] Log health data
    s.add(
        actions.make(
            "health.quantity.log",
            WFQuantitySampleType="Toothbrushing",
            WFQuantitySampleDate=menu_result,
            WFSampleEndDate=adjusted,
        )
    )

    return s


# ============================================================
# Test 3: Golden Hour Times (15 actions, nested repeat-each)
# ============================================================


def reconstruct_golden_hour():
    """Reconstruct Check_upcoming_Golden_Hour_times.shortcut.

    Tests: nested repeat_each blocks, properties actions, date adjustment,
    calendar events, complex parameter wiring.

    Note: Some parameters use advanced features (Aggrandizements, WFTimeOffsetValue,
    calendar descriptors) that require pre-wrapped dicts. The test verifies
    structural correctness (action sequence, param keys, control flow) while
    using pre-wrapped values from the original for complex parameters.
    """
    orig = load_original("Check_upcoming_Golden_Hour_times.shortcut")

    s = Shortcut("Check upcoming Golden Hour times")

    # [0] Comment
    s.add(
        actions.make(
            "comment",
            WFCommentActionText=orig[0]["WFWorkflowActionParameters"][
                "WFCommentActionText"
            ],
        )
    )

    # [1] Get weather forecast
    forecast = s.add(actions.make("weather.forecast"))

    # [2-4] Repeat with each day in forecast
    with s.repeat_each_block(forecast):
        # [3] Set item name — uses Repeat Item variable with Aggrandizements.
        # Pass the complex WFName value from original since it uses
        # property access + custom date formatting beyond simple ActionHandle.
        s.add(
            actions.make(
                "setitemname",
                WFName=orig[3]["WFWorkflowActionParameters"]["WFName"],
                WFInput=wrap_token_attachment(ref_variable("Repeat Item")),
            )
        )

    # [5] Choose from list (multi-select) — references Repeat Results
    # The repeat end action output is the last action before this
    repeat_end_uuid = s.actions[-1]["WFWorkflowActionParameters"].get("UUID")
    if not repeat_end_uuid:
        import uuid as uuid_mod

        repeat_end_uuid = str(uuid_mod.uuid4()).upper()
        s.actions[-1]["WFWorkflowActionParameters"]["UUID"] = repeat_end_uuid
    repeat_results = ActionHandle(
        repeat_end_uuid, "Repeat Results", "is.workflow.actions.repeat.each"
    )

    chosen_days = s.add(
        actions.make(
            "choosefromlist",
            WFChooseFromListActionPrompt="Which days?",
            WFChooseFromListActionSelectMultiple=True,
            WFInput=repeat_results,
        )
    )

    # [6-14] Second repeat: for each chosen day
    with s.repeat_each_block(chosen_days):
        # [7] Get sunset time — uses Repeat Item variable
        sunset = s.add(
            actions.make(
                "properties.weather.conditions",
                WFContentItemPropertyName="Sunset Time",
                WFInput=wrap_token_attachment(ref_variable("Repeat Item")),
                CustomOutputName="Sunset",
            )
        )

        # [8] Adjust date: subtract 1 hour from sunset
        golden_start = s.add(
            actions.make(
                "adjustdate",
                WFDate=sunset,
                WFDuration=actions.build_quantity(1, "hr"),
                CustomOutputName="Sunset-1",
                WFAdjustOffsetPicker=orig[8]["WFWorkflowActionParameters"][
                    "WFAdjustOffsetPicker"
                ],
                WFAdjustOperation="Subtract",
            )
        )

        # [9] Add calendar event: Golden Hour
        s.add(
            actions.make(
                "addnewevent",
                WFCalendarItemTitle="Golden hour",
                WFCalendarDescriptor=orig[9]["WFWorkflowActionParameters"][
                    "WFCalendarDescriptor"
                ],
                WFAlertTime="1 hour before",
                WFCalendarItemDates=True,
                WFCalendarItemEndDate=sunset,
                ShowWhenRun=False,
                WFCalendarItemStartDate=golden_start,
                WFCalendarItemCalendar="Photography",
            )
        )

        # [10] Get sunrise time
        sunrise = s.add(
            actions.make(
                "properties.weather.conditions",
                WFContentItemPropertyName="Sunrise Time",
                WFInput=wrap_token_attachment(ref_variable("Repeat Item")),
                CustomOutputName="Sunrise Hour",
            )
        )

        # [11] Adjust date: add 1 hour to sunrise
        sunrise_plus1 = s.add(
            actions.make(
                "adjustdate",
                WFDate=sunrise,
                WFDuration=actions.build_quantity(1, "hr"),
                CustomOutputName="Sunrise Hour +1",
                WFAdjustOffsetPicker=orig[11]["WFWorkflowActionParameters"][
                    "WFAdjustOffsetPicker"
                ],
                WFAdjustOperation="Add",
            )
        )

        # [12] Adjust date: subtract 30 min from sunrise+1
        half_through = s.add(
            actions.make(
                "adjustdate",
                WFDate=sunrise_plus1,
                WFDuration=actions.build_quantity(30, "min"),
                CustomOutputName="Half Through",
                WFAdjustOffsetPicker=orig[12]["WFWorkflowActionParameters"][
                    "WFAdjustOffsetPicker"
                ],
                WFAdjustOperation="Subtract",
            )
        )

        # [13] Add calendar event: Golden hour (morning)
        s.add(
            actions.make(
                "addnewevent",
                WFCalendarItemTitle="Golden hour",
                WFCalendarDescriptor=orig[13]["WFWorkflowActionParameters"][
                    "WFCalendarDescriptor"
                ],
                WFAlertTime="Custom",
                WFCalendarItemDates=True,
                WFCalendarItemEndDate=sunrise_plus1,
                ShowWhenRun=False,
                WFAlertCustomTime=half_through,
                WFCalendarItemStartDate=sunrise,
                WFCalendarItemCalendar="Photography",
            )
        )

    return s


# ============================================================
# Run all tests
# ============================================================


def run_roundtrip(name, filename, reconstruct_fn):
    global passed, failed
    print(f"\n{'=' * 60}")
    print(f"Test: {name}")
    print(f"{'=' * 60}")

    try:
        original = load_original(filename)
        reconstructed_shortcut = reconstruct_fn()
        reconstructed = reconstructed_shortcut.actions

        checks_passed, checks_failed, details = compare_actions(original, reconstructed)

        for d in details:
            print(f"  ! {d}")

        if checks_failed == 0:
            print(f"  PASS: {checks_passed} checks passed")
            passed += 1
        else:
            print(f"  FAIL: {checks_passed} passed, {checks_failed} failed")
            failed += 1

    except Exception as e:
        import traceback

        print(f"  FAIL: Exception during reconstruction: {e}")
        traceback.print_exc()
        failed += 1


if __name__ == "__main__":
    print("Apple Shortcuts Compiler \u2014 Round-Trip Reconstruction Tests")

    run_roundtrip(
        "Find Gas Nearby (5 actions, linear)",
        "Find_gas_nearby.shortcut",
        reconstruct_find_gas,
    )

    run_roundtrip(
        "Log Toothbrushing (9 actions, menu)",
        "Log_toothbrushing.shortcut",
        reconstruct_log_toothbrushing,
    )

    run_roundtrip(
        "Golden Hour Times (15 actions, nested repeat)",
        "Check_upcoming_Golden_Hour_times.shortcut",
        reconstruct_golden_hour,
    )

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("All round-trip tests passed.")
