#!/usr/bin/env python3
"""
Scan all .shortcut files for control flow patterns.
"""

import plistlib
from pathlib import Path
from collections import defaultdict

# Configuration (relative to the project root)
BASE_DIR = Path(__file__).resolve().parent.parent
SHORTCUTS_DIR = BASE_DIR / "downloaded"

# Control flow action identifiers we're tracking
CONTROL_FLOW_ACTIONS = {
    'is.workflow.actions.conditional': 'if/else',
    'is.workflow.actions.choosefrommenu': 'menu',
    'is.workflow.actions.repeat.count': 'repeat N times',
    'is.workflow.actions.repeat.each': 'repeat with each',
}

# Flow-related actions
OTHER_FLOW_ACTIONS = {
    'is.workflow.actions.output': 'output/stop',
    'is.workflow.actions.nothing': 'nothing',
}


def scan_shortcut_file(file_path):
    """Parse a .shortcut file and extract control flow information."""
    try:
        with open(file_path, 'rb') as f:
            plist = plistlib.load(f)
        return plist
    except Exception as e:
        return None


def analyze_control_flow(plist):
    """Extract control flow information from a plist."""
    results = {
        'control_flow_actions': [],
        'other_flow_actions': [],
        'unknown_control_flow': [],
        'control_flow_by_type': defaultdict(list),
    }
    
    if not isinstance(plist, dict) or 'WFWorkflowActions' not in plist:
        return results
    
    actions = plist['WFWorkflowActions']
    if not isinstance(actions, list):
        return results
    
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        
        action_id = action.get('WFWorkflowActionIdentifier', '')
        params = action.get('WFWorkflowActionParameters', {})
        if not isinstance(params, dict):
            params = {}
        
        control_flow_mode = params.get('WFControlFlowMode')
        grouping_id = params.get('GroupingIdentifier')
        
        # Check if this is a known control flow action
        if action_id in CONTROL_FLOW_ACTIONS:
            results['control_flow_actions'].append({
                'index': idx,
                'type': CONTROL_FLOW_ACTIONS[action_id],
                'action_id': action_id,
                'control_flow_mode': control_flow_mode,
                'grouping_id': grouping_id,
                'has_grouping': grouping_id is not None and grouping_id != '',
            })
            
            results['control_flow_by_type'][action_id].append({
                'mode': control_flow_mode,
                'grouping_id': grouping_id,
            })
        
        # Check if this is another flow-related action
        elif action_id in OTHER_FLOW_ACTIONS:
            results['other_flow_actions'].append({
                'index': idx,
                'type': OTHER_FLOW_ACTIONS[action_id],
                'action_id': action_id,
            })
        
        # Check for any action with WFControlFlowMode set (shouldn't be there)
        elif 'WFControlFlowMode' in params:
            control_flow_mode = params.get('WFControlFlowMode')
            results['unknown_control_flow'].append({
                'index': idx,
                'action_id': action_id,
                'control_flow_mode': control_flow_mode,
                'grouping_id': params.get('GroupingIdentifier'),
            })
        
        # Check for actions with "flow" or "control" in identifier
        elif 'flow' in action_id.lower() or 'control' in action_id.lower():
            results['other_flow_actions'].append({
                'index': idx,
                'type': 'flow/control related',
                'action_id': action_id,
            })
    
    return results


def main():
    print("=" * 80)
    print("SHORTCUT CONTROL FLOW PATTERN SCANNER")
    print("=" * 80)
    print()
    
    # Find all .shortcut files
    shortcuts_path = Path(SHORTCUTS_DIR)
    shortcut_files = sorted(shortcuts_path.glob('*.shortcut'))
    
    print(f"Found {len(shortcut_files)} shortcut files")
    print()
    
    # Statistics
    stats = {
        'total_shortcuts': len(shortcut_files),
        'shortcuts_with_control_flow': 0,
        'shortcuts_with_conditional': 0,
        'shortcuts_with_menu': 0,
        'shortcuts_with_repeat_count': 0,
        'shortcuts_with_repeat_each': 0,
        'control_flow_action_counts': defaultdict(int),
        'control_flow_by_mode': defaultdict(lambda: defaultdict(int)),
        'all_modes': defaultdict(int),
        'unusual_modes': defaultdict(int),
        'missing_grouping': [],
        'unknown_control_flow_actions': [],
    }
    
    # Process each shortcut
    processed = 0
    for shortcut_file in shortcut_files:
        processed += 1
        if processed % 50 == 0:
            print(f"Processing... {processed}/{len(shortcut_files)}")
        
        plist = scan_shortcut_file(shortcut_file)
        if plist is None:
            continue
        
        analysis = analyze_control_flow(plist)
        
        # Track which shortcuts have control flow
        if analysis['control_flow_actions']:
            stats['shortcuts_with_control_flow'] += 1
        
        # Track action types
        for cf_action in analysis['control_flow_actions']:
            action_type = cf_action['type']
            action_id = cf_action['action_id']
            mode = cf_action['control_flow_mode']
            has_grouping = cf_action['has_grouping']
            
            stats['control_flow_action_counts'][action_type] += 1
            stats['control_flow_by_mode'][action_type][mode] += 1
            stats['all_modes'][mode] += 1
            
            # Check for unusual modes
            if mode not in [0, 1, 2]:
                stats['unusual_modes'][mode] += 1
            
            # Check for missing grouping
            if not has_grouping:
                stats['missing_grouping'].append({
                    'file': shortcut_file.name,
                    'type': action_type,
                    'mode': mode,
                })
        
        # Track unknown control flow
        for unknown in analysis['unknown_control_flow']:
            stats['unknown_control_flow_actions'].append({
                'file': shortcut_file.name,
                'action_id': unknown['action_id'],
                'mode': unknown['control_flow_mode'],
            })
    
    # Count shortcuts with each type separately
    shortcuts_by_type = defaultdict(set)
    for shortcut_file in shortcut_files:
        plist = scan_shortcut_file(shortcut_file)
        if plist is None:
            continue
        analysis = analyze_control_flow(plist)
        for cf_action in analysis['control_flow_actions']:
            action_id = cf_action['action_id']
            if action_id == 'is.workflow.actions.conditional':
                shortcuts_by_type['conditional'].add(shortcut_file.name)
            elif action_id == 'is.workflow.actions.choosefrommenu':
                shortcuts_by_type['menu'].add(shortcut_file.name)
            elif action_id == 'is.workflow.actions.repeat.count':
                shortcuts_by_type['repeat_count'].add(shortcut_file.name)
            elif action_id == 'is.workflow.actions.repeat.each':
                shortcuts_by_type['repeat_each'].add(shortcut_file.name)
    
    print()
    print("=" * 80)
    print("CONTROL FLOW STATISTICS")
    print("=" * 80)
    print()
    
    # Report 1: How many shortcuts use each control flow type
    print("1. SHORTCUT USAGE BY CONTROL FLOW TYPE:")
    print("-" * 80)
    print(f"   Total shortcuts scanned: {stats['total_shortcuts']}")
    print(f"   Shortcuts with ANY control flow: {stats['shortcuts_with_control_flow']}")
    print(f"   Shortcuts with if/else (conditional): {len(shortcuts_by_type['conditional'])}")
    print(f"   Shortcuts with menu (choosefrommenu): {len(shortcuts_by_type['menu'])}")
    print(f"   Shortcuts with repeat N times: {len(shortcuts_by_type['repeat_count'])}")
    print(f"   Shortcuts with repeat with each: {len(shortcuts_by_type['repeat_each'])}")
    print()
    
    # Report 2: Total count of each control flow action by mode
    print("2. CONTROL FLOW ACTION COUNTS BY MODE:")
    print("-" * 80)
    for action_type in CONTROL_FLOW_ACTIONS.values():
        total = stats['control_flow_action_counts'][action_type]
        mode_0 = stats['control_flow_by_mode'][action_type].get(0, 0)
        mode_1 = stats['control_flow_by_mode'][action_type].get(1, 0)
        mode_2 = stats['control_flow_by_mode'][action_type].get(2, 0)
        
        print(f"   {action_type}:")
        print(f"      Total: {total}")
        print(f"      Mode 0 (block start): {mode_0}")
        print(f"      Mode 1 (middle): {mode_1}")
        print(f"      Mode 2 (block end): {mode_2}")
        print()
    
    # Report overall by mode
    print("   TOTAL BY MODE (all control flow actions):")
    mode_0_total = stats['all_modes'].get(0, 0)
    mode_1_total = stats['all_modes'].get(1, 0)
    mode_2_total = stats['all_modes'].get(2, 0)
    other_modes = sum(v for k, v in stats['all_modes'].items() if k not in [0, 1, 2])
    print(f"      Mode 0 (block start): {mode_0_total}")
    print(f"      Mode 1 (middle): {mode_1_total}")
    print(f"      Mode 2 (block end): {mode_2_total}")
    print(f"      Other/None: {other_modes}")
    print()
    
    # Report 3: Block nesting analysis
    print("3. BLOCK NESTING ANALYSIS (Mode 0 vs Mode 2):")
    print("-" * 80)
    if mode_0_total == 0 and mode_2_total == 0:
        print(f"   NOTE: No traditional block start/end modes found (all actions use mode None)")
        print(f"   This suggests a different control flow structure than expected.")
        print(f"   Total actions with mode None or missing: {other_modes}")
    elif mode_0_total == mode_2_total:
        print(f"   ✓ VALID: Block starts (mode 0) = Block ends (mode 2) = {mode_0_total}")
        print(f"   All blocks appear to be properly closed!")
    else:
        print(f"   ✗ MISMATCH: Block starts (mode 0) = {mode_0_total}, Block ends (mode 2) = {mode_2_total}")
        print(f"   Difference: {abs(mode_0_total - mode_2_total)} blocks")
        if mode_0_total > mode_2_total:
            print(f"   WARNING: More block starts than ends - some blocks may not be closed!")
        else:
            print(f"   WARNING: More block ends than starts - structure may be malformed!")
    print()
    
    # Report 4: Unusual modes
    print("4. UNUSUAL WFControlFlowMode VALUES:")
    print("-" * 80)
    if stats['unusual_modes']:
        print(f"   Found unusual mode values:")
        for mode, count in sorted(stats['unusual_modes'].items()):
            print(f"      Mode {mode}: {count} occurrences")
    
    # Count None/missing modes
    none_count = stats['all_modes'].get(None, 0)
    if none_count > 0:
        print(f"      Mode None (missing/null): {none_count} occurrences")
    
    if not stats['unusual_modes'] and none_count == 0:
        print(f"   ✓ Only modes 0, 1, and 2 used")
    print()
    
    # Report 5: GroupingIdentifier coverage
    print("5. GroupingIdentifier COVERAGE:")
    print("-" * 80)
    total_control_flow = sum(stats['control_flow_action_counts'].values())
    missing = len(stats['missing_grouping'])
    coverage_pct = ((total_control_flow - missing) / total_control_flow * 100) if total_control_flow > 0 else 0
    print(f"   Total control flow actions: {total_control_flow}")
    print(f"   With GroupingIdentifier: {total_control_flow - missing}")
    print(f"   Missing GroupingIdentifier: {missing}")
    print(f"   Coverage: {coverage_pct:.1f}%")
    
    if missing > 0 and missing <= 20:
        print(f"\n   Actions missing GroupingIdentifier:")
        for i, item in enumerate(stats['missing_grouping']):
            print(f"      {i+1}. {item['file']} - {item['type']} (mode {item['mode']})")
    elif missing > 20:
        print(f"\n   First 10 actions missing GroupingIdentifier:")
        for i, item in enumerate(stats['missing_grouping'][:10]):
            print(f"      {i+1}. {item['file']} - {item['type']} (mode {item['mode']})")
        print(f"      ... and {missing - 10} more")
    print()
    
    # Report 6: Other control flow actions
    print("6. OTHER CONTROL FLOW ACTIONS:")
    print("-" * 80)
    
    # Recount other flow actions
    other_flow_counts = defaultdict(int)
    for shortcut_file in shortcut_files:
        plist = scan_shortcut_file(shortcut_file)
        if plist is None:
            continue
        analysis = analyze_control_flow(plist)
        for action in analysis['other_flow_actions']:
            other_flow_counts[action['type']] += 1
    
    if other_flow_counts:
        for action_type in sorted(other_flow_counts.keys()):
            count = other_flow_counts[action_type]
            print(f"   {action_type}: {count}")
    else:
        print(f"   None found")
    print()
    
    # Report 7: Unknown control flow actions
    print("7. UNKNOWN CONTROL FLOW ACTIONS:")
    print("-" * 80)
    if stats['unknown_control_flow_actions']:
        print(f"   Found {len(stats['unknown_control_flow_actions'])} actions with WFControlFlowMode set")
        print(f"   (These are actions not in our known control flow list):")
        print()
        unknown_by_id = defaultdict(list)
        for item in stats['unknown_control_flow_actions']:
            unknown_by_id[item['action_id']].append(item)
        
        for action_id in sorted(unknown_by_id.keys()):
            items = unknown_by_id[action_id]
            print(f"   {action_id}:")
            print(f"      Count: {len(items)}")
            modes = defaultdict(int)
            for item in items:
                modes[item['mode']] += 1
            for mode in sorted(modes.keys()):
                print(f"      Mode {mode}: {modes[mode]}")
            print()
    else:
        print(f"   ✓ No unknown control flow actions found")
    print()
    
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
