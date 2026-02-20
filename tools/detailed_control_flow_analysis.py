#!/usr/bin/env python3
"""
Detailed control flow analysis - find the block mismatch and identify flow-related actions.
"""

import plistlib
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SHORTCUTS_DIR = BASE_DIR / "downloaded"

CONTROL_FLOW_ACTIONS = {
    "is.workflow.actions.conditional": "if/else",
    "is.workflow.actions.choosefrommenu": "menu",
    "is.workflow.actions.repeat.count": "repeat N times",
    "is.workflow.actions.repeat.each": "repeat with each",
}


def scan_shortcut_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return plistlib.load(f)
    except Exception:
        return None


def main():
    print("=" * 80)
    print("DETAILED CONTROL FLOW ANALYSIS")
    print("=" * 80)
    print()

    shortcuts_path = Path(SHORTCUTS_DIR)
    shortcut_files = sorted(shortcuts_path.glob("*.shortcut"))

    # Collect all flow-related actions
    flow_actions = defaultdict(int)
    flow_actions_detailed = defaultdict(list)

    # Track block starts and ends for analysis
    block_analysis = []

    # Process all shortcuts
    for shortcut_file in shortcut_files:
        plist = scan_shortcut_file(shortcut_file)
        if plist is None or "WFWorkflowActions" not in plist:
            continue

        actions = plist["WFWorkflowActions"]
        if not isinstance(actions, list):
            continue

        mode_0_count = 0
        mode_2_count = 0

        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue

            action_id = action.get("WFWorkflowActionIdentifier", "")
            params = action.get("WFWorkflowActionParameters", {})
            if not isinstance(params, dict):
                params = {}

            # Check for any action with "flow" or "control" in the identifier
            if "flow" in action_id.lower() or "control" in action_id.lower():
                flow_actions[action_id] += 1
                flow_actions_detailed[action_id].append((shortcut_file.name, idx))

            # Track control flow blocks
            if action_id in CONTROL_FLOW_ACTIONS:
                mode = params.get("WFControlFlowMode")
                if mode == 0:
                    mode_0_count += 1
                elif mode == 2:
                    mode_2_count += 1

        if mode_0_count != mode_2_count:
            block_analysis.append(
                {
                    "file": shortcut_file.name,
                    "mode_0": mode_0_count,
                    "mode_2": mode_2_count,
                }
            )

    # Report 1: All flow-related actions found
    print("1. ALL FLOW/CONTROL-RELATED ACTIONS FOUND:")
    print("-" * 80)

    total_flow = 0
    for action_id in sorted(flow_actions.keys()):
        count = flow_actions[action_id]
        total_flow += count
        print(f"   {action_id}: {count}")

    print(f"\n   TOTAL: {total_flow}")
    print()

    # Report 2: Shortcuts with block start/end mismatch
    print("2. SHORTCUTS WITH BLOCK START/END MISMATCH:")
    print("-" * 80)

    if block_analysis:
        print(f"   Found {len(block_analysis)} shortcuts with mismatches:")
        print()
        for item in block_analysis:
            if item["mode_0"] > item["mode_2"]:
                status = f"More starts ({item['mode_0']} vs {item['mode_2']})"
            else:
                status = f"More ends ({item['mode_2']} vs {item['mode_0']})"
            print(f"   {item['file']}: {status}")
    else:
        print("   All shortcuts have balanced block starts and ends!")
    print()

    # Report 3: Sample of flow-related actions by type
    print("3. FLOW-RELATED ACTIONS SAMPLE (up to 3 per action type):")
    print("-" * 80)

    for action_id in sorted(flow_actions_detailed.keys()):
        items = flow_actions_detailed[action_id]
        print(f"\n   {action_id}:")
        for i, (filename, idx) in enumerate(items[:3]):
            print(f"      {i + 1}. {filename} (action index {idx})")
        if len(items) > 3:
            print(f"      ... and {len(items) - 3} more")
    print()

    print("=" * 80)
    print("DETAILED ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
