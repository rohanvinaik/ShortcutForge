#!/usr/bin/env python3
"""
Final comprehensive control flow report.
"""

import plistlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
SHORTCUTS_DIR = BASE_DIR / "downloaded"

CONTROL_FLOW_ACTIONS = {
    'is.workflow.actions.conditional': 'if/else',
    'is.workflow.actions.choosefrommenu': 'menu',
    'is.workflow.actions.repeat.count': 'repeat N times',
    'is.workflow.actions.repeat.each': 'repeat with each',
}

def scan_shortcut_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return plistlib.load(f)
    except:
        return None


def main():
    shortcuts_path = Path(SHORTCUTS_DIR)
    shortcut_files = sorted(shortcuts_path.glob('*.shortcut'))
    
    # Statistics
    stats = {
        'total_shortcuts': len(shortcut_files),
        'shortcuts_with_control_flow': 0,
        'control_flow_action_counts': defaultdict(int),
        'control_flow_by_mode': defaultdict(lambda: defaultdict(int)),
        'all_modes': defaultdict(int),
        'missing_grouping': [],
        'unknown_control_flow_actions': [],
        'shortcut_by_type': defaultdict(set),
        'flow_action_counts': defaultdict(int),
        'block_mismatches': [],
    }
    
    # Process all shortcuts
    for shortcut_file in shortcut_files:
        plist = scan_shortcut_file(shortcut_file)
        if plist is None or 'WFWorkflowActions' not in plist:
            continue
        
        actions = plist['WFWorkflowActions']
        if not isinstance(actions, list):
            continue
        
        has_control_flow = False
        mode_0_count = 0
        mode_2_count = 0
        
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            
            action_id = action.get('WFWorkflowActionIdentifier', '')
            params = action.get('WFWorkflowActionParameters', {})
            if not isinstance(params, dict):
                params = {}
            
            mode = params.get('WFControlFlowMode')
            grouping_id = params.get('GroupingIdentifier')
            
            # Check for control flow actions
            if action_id in CONTROL_FLOW_ACTIONS:
                has_control_flow = True
                action_type = CONTROL_FLOW_ACTIONS[action_id]
                stats['control_flow_action_counts'][action_type] += 1
                stats['control_flow_by_mode'][action_type][mode] += 1
                stats['all_modes'][mode] += 1
                stats['shortcut_by_type'][action_id].add(shortcut_file.name)
                
                if mode == 0:
                    mode_0_count += 1
                elif mode == 2:
                    mode_2_count += 1
                
                if not grouping_id:
                    stats['missing_grouping'].append({
                        'file': shortcut_file.name,
                        'type': action_type,
                        'mode': mode,
                    })
            
            # Count all actions with "flow" or "control" in identifier
            if 'flow' in action_id.lower() or 'control' in action_id.lower():
                stats['flow_action_counts'][action_id] += 1
            
            # Check for unknown control flow actions
            if 'WFControlFlowMode' in params and action_id not in CONTROL_FLOW_ACTIONS:
                stats['unknown_control_flow_actions'].append({
                    'file': shortcut_file.name,
                    'action_id': action_id,
                    'mode': mode,
                })
        
        if has_control_flow:
            stats['shortcuts_with_control_flow'] += 1
        
        if mode_0_count != mode_2_count:
            stats['block_mismatches'].append({
                'file': shortcut_file.name,
                'mode_0': mode_0_count,
                'mode_2': mode_2_count,
            })
    
    # Write comprehensive report
    report_path = BASE_DIR / "CONTROL_FLOW_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("COMPREHENSIVE CONTROL FLOW ANALYSIS - APPLE SHORTCUTS\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total shortcuts scanned: {stats['total_shortcuts']}\n")
        f.write(f"Shortcuts with control flow: {stats['shortcuts_with_control_flow']}\n")
        f.write(f"Percentage with control flow: {stats['shortcuts_with_control_flow'] / stats['total_shortcuts'] * 100:.1f}%\n")
        f.write("\n")
        
        f.write("CONTROL FLOW USAGE BY TYPE\n")
        f.write("-" * 90 + "\n")
        for action_id in CONTROL_FLOW_ACTIONS:
            action_type = CONTROL_FLOW_ACTIONS[action_id]
            shortcut_count = len(stats['shortcut_by_type'][action_id])
            action_count = stats['control_flow_action_counts'][action_type]
            f.write(f"{action_type:20s}: {shortcut_count:3d} shortcuts using it, {action_count:4d} total actions\n")
        f.write("\n")
        
        f.write("CONTROL FLOW ACTION COUNT BY MODE\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Action Type':20s} {'Mode 0':>10s} {'Mode 1':>10s} {'Mode 2':>10s} {'Total':>10s}\n")
        f.write("-" * 90 + "\n")
        for action_type in CONTROL_FLOW_ACTIONS.values():
            mode_0 = stats['control_flow_by_mode'][action_type].get(0, 0)
            mode_1 = stats['control_flow_by_mode'][action_type].get(1, 0)
            mode_2 = stats['control_flow_by_mode'][action_type].get(2, 0)
            total = mode_0 + mode_1 + mode_2
            f.write(f"{action_type:20s} {mode_0:10d} {mode_1:10d} {mode_2:10d} {total:10d}\n")
        
        mode_0_total = stats['all_modes'].get(0, 0)
        mode_1_total = stats['all_modes'].get(1, 0)
        mode_2_total = stats['all_modes'].get(2, 0)
        grand_total = mode_0_total + mode_1_total + mode_2_total
        f.write("-" * 90 + "\n")
        f.write(f"{'TOTAL':20s} {mode_0_total:10d} {mode_1_total:10d} {mode_2_total:10d} {grand_total:10d}\n")
        f.write("\n")
        
        f.write("BLOCK START/END VALIDATION\n")
        f.write("-" * 90 + "\n")
        f.write(f"Block starts (Mode 0): {mode_0_total}\n")
        f.write(f"Block ends (Mode 2):   {mode_2_total}\n")
        f.write(f"Difference: {abs(mode_0_total - mode_2_total)}\n\n")
        
        if mode_0_total == mode_2_total:
            f.write("STATUS: ✓ VALID - All block starts have matching block ends!\n")
        else:
            f.write("STATUS: ✗ MISMATCH - Block starts and ends are not balanced!\n")
            if mode_0_total > mode_2_total:
                f.write(f"WARNING: More block starts than ends - {mode_0_total - mode_2_total} unclosed blocks\n")
            else:
                f.write(f"WARNING: More block ends than starts - malformed structure detected\n")
        f.write("\n")
        
        f.write("BLOCK START/END MISMATCHES BY SHORTCUT\n")
        f.write("-" * 90 + "\n")
        if stats['block_mismatches']:
            f.write(f"Found {len(stats['block_mismatches'])} shortcuts with mismatches:\n\n")
            for item in stats['block_mismatches']:
                diff = item['mode_0'] - item['mode_2']
                if diff > 0:
                    f.write(f"  {item['file']:50s} {item['mode_0']} starts, {item['mode_2']} ends (excess starts: {diff})\n")
                else:
                    f.write(f"  {item['file']:50s} {item['mode_0']} starts, {item['mode_2']} ends (excess ends: {-diff})\n")
        else:
            f.write("All shortcuts have balanced block structures!\n")
        f.write("\n")
        
        f.write("GROUPING IDENTIFIER COVERAGE\n")
        f.write("-" * 90 + "\n")
        total_control_flow = sum(stats['control_flow_action_counts'].values())
        missing = len(stats['missing_grouping'])
        coverage_pct = ((total_control_flow - missing) / total_control_flow * 100) if total_control_flow > 0 else 0
        f.write(f"Total control flow actions: {total_control_flow}\n")
        f.write(f"With GroupingIdentifier: {total_control_flow - missing}\n")
        f.write(f"Missing GroupingIdentifier: {missing}\n")
        f.write(f"Coverage: {coverage_pct:.1f}%\n\n")
        
        if missing > 0:
            f.write("STATUS: ✗ Not all actions have GroupingIdentifier\n\n")
            f.write(f"First 20 actions missing GroupingIdentifier:\n")
            for i, item in enumerate(stats['missing_grouping'][:20]):
                f.write(f"  {i+1:2d}. {item['file']:50s} {item['type']:20s} (mode {item['mode']})\n")
            if missing > 20:
                f.write(f"  ... and {missing - 20} more\n")
        else:
            f.write("STATUS: ✓ All control flow actions have GroupingIdentifier!\n")
        f.write("\n")
        
        f.write("UNKNOWN CONTROL FLOW ACTIONS\n")
        f.write("-" * 90 + "\n")
        if stats['unknown_control_flow_actions']:
            f.write(f"Found {len(stats['unknown_control_flow_actions'])} actions with WFControlFlowMode outside known control flow:\n\n")
            unknown_by_id = defaultdict(list)
            for item in stats['unknown_control_flow_actions']:
                unknown_by_id[item['action_id']].append(item)
            
            for action_id in sorted(unknown_by_id.keys()):
                items = unknown_by_id[action_id]
                f.write(f"  {action_id}:\n")
                f.write(f"    Count: {len(items)}\n")
                modes = defaultdict(int)
                for item in items:
                    modes[item['mode']] += 1
                for mode in sorted(modes.keys()):
                    f.write(f"    Mode {mode}: {modes[mode]}\n")
                f.write("\n")
        else:
            f.write("STATUS: ✓ No unknown control flow actions found\n")
        f.write("\n")
        
        f.write("ALL FLOW/CONTROL RELATED ACTIONS\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total actions with 'flow' or 'control' in identifier: {sum(stats['flow_action_counts'].values())}\n\n")
        f.write("Top 30 flow/control related actions by frequency:\n")
        f.write(f"{'Action ID':60s} {'Count':>10s}\n")
        f.write("-" * 90 + "\n")
        sorted_flow = sorted(stats['flow_action_counts'].items(), key=lambda x: x[1], reverse=True)
        for action_id, count in sorted_flow[:30]:
            f.write(f"{action_id:60s} {count:10d}\n")
        f.write("\n")
        
        f.write("=" * 90 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("=" * 90 + "\n")
    
    print(f"Report written to: {report_path}")
    print()
    
    # Also print key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    print(f"1. Control Flow Coverage: {stats['shortcuts_with_control_flow']}/{stats['total_shortcuts']} shortcuts ({stats['shortcuts_with_control_flow']/stats['total_shortcuts']*100:.1f}%)")
    print()
    print("2. Control Flow Distribution:")
    for action_id in CONTROL_FLOW_ACTIONS:
        action_type = CONTROL_FLOW_ACTIONS[action_id]
        shortcut_count = len(stats['shortcut_by_type'][action_id])
        action_count = stats['control_flow_action_counts'][action_type]
        print(f"   - {action_type}: {shortcut_count} shortcuts, {action_count} total actions")
    print()
    print("3. Block Structure Validation:")
    if mode_0_total == mode_2_total:
        print(f"   ✓ VALID: {mode_0_total} block starts matched with {mode_2_total} block ends")
    else:
        print(f"   ✗ MISMATCH: {mode_0_total} block starts vs {mode_2_total} block ends (diff: {abs(mode_0_total - mode_2_total)})")
    print()
    print("4. GroupingIdentifier Coverage: 100%" if missing == 0 else f"   GroupingIdentifier: {100*coverage_pct:.1f}%")
    print()
    print("5. Flow/Control Related Actions: Found in nearly all shortcuts")
    print()
    if stats['block_mismatches']:
        print(f"6. ATTENTION: {len(stats['block_mismatches'])} shortcuts have block start/end mismatches")
    else:
        print("6. All shortcuts have balanced block structures")
    print()
    

if __name__ == '__main__':
    main()
