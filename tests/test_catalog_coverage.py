"""
Catalog Coverage Stress Test
=============================
Load all downloaded shortcuts and verify the compiler's catalog
covers every action identifier and parameter name found in real data.

Run: python3 scripts/test_catalog_coverage.py
"""
import plistlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
import shortcuts_compiler
from shortcuts_compiler import _resolve_identifier, _load_catalog

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "downloaded")

# Force catalog load
_load_catalog()
_ACTION_CATALOG = shortcuts_compiler._ACTION_CATALOG

# Parameters injected by the compiler — always valid, never user-specified
COMPILER_PARAMS = {"UUID", "CustomOutputName", "GroupingIdentifier", "WFControlFlowMode"}

results = {
    "shortcuts_processed": 0,
    "total_actions": 0,
    "resolved_actions": 0,
    "cataloged_actions": 0,
    "unresolved_identifiers": {},   # identifier -> count
    "uncataloged_identifiers": {},  # identifier -> count (resolves but no catalog entry)
    "unknown_params": {},           # identifier -> {param_name: count}
    "total_param_instances": 0,
    "covered_param_instances": 0,
}

for fname in sorted(os.listdir(DOWNLOAD_DIR)):
    if not fname.endswith(".shortcut"):
        continue
    fpath = os.path.join(DOWNLOAD_DIR, fname)
    try:
        with open(fpath, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        continue

    results["shortcuts_processed"] += 1
    actions_list = plist.get("WFWorkflowActions", [])

    for action in actions_list:
        identifier = action.get("WFWorkflowActionIdentifier", "")
        if not identifier:
            continue
        results["total_actions"] += 1

        # Test 1: Can we resolve this identifier?
        try:
            resolved = _resolve_identifier(identifier)
            results["resolved_actions"] += 1
        except ValueError:
            results["unresolved_identifiers"][identifier] = \
                results["unresolved_identifiers"].get(identifier, 0) + 1
            continue

        # Test 2: Does the catalog have an entry with observed_params?
        catalog_entry = _ACTION_CATALOG.get(resolved, {})
        if catalog_entry:
            results["cataloged_actions"] += 1
        else:
            results["uncataloged_identifiers"][resolved] = \
                results["uncataloged_identifiers"].get(resolved, 0) + 1

        # Test 3: Are all parameters in the catalog's observed_params?
        params = action.get("WFWorkflowActionParameters", {})
        known_params = set(catalog_entry.get("observed_params", {}).keys()) | COMPILER_PARAMS

        for param_name in params:
            results["total_param_instances"] += 1
            if param_name in known_params:
                results["covered_param_instances"] += 1
            else:
                if resolved not in results["unknown_params"]:
                    results["unknown_params"][resolved] = {}
                results["unknown_params"][resolved][param_name] = \
                    results["unknown_params"][resolved].get(param_name, 0) + 1


# === Report ===
print("=" * 70)
print("CATALOG COVERAGE STRESS TEST")
print("=" * 70)

total = results["total_actions"]
resolved = results["resolved_actions"]
cataloged = results["cataloged_actions"]
covered = results["covered_param_instances"]
total_params = results["total_param_instances"]

print(f"Shortcuts processed:  {results['shortcuts_processed']}")
print(f"Total actions:        {total}")
print()
print(f"Action resolution:    {resolved}/{total} ({100*resolved/max(1,total):.1f}%)")
print(f"Cataloged actions:    {cataloged}/{total} ({100*cataloged/max(1,total):.1f}%)")
print(f"Parameter coverage:   {covered}/{total_params} ({100*covered/max(1,total_params):.1f}%)")
print()

unresolved = results["unresolved_identifiers"]
if unresolved:
    print(f"UNRESOLVED IDENTIFIERS ({len(unresolved)} unique):")
    for ident, count in sorted(unresolved.items(), key=lambda x: -x[1])[:20]:
        print(f"  {ident:60s} ({count}x)")
    print()

uncataloged = results["uncataloged_identifiers"]
if uncataloged:
    print(f"RESOLVED BUT UNCATALOGED ({len(uncataloged)} unique):")
    for ident, count in sorted(uncataloged.items(), key=lambda x: -x[1])[:20]:
        print(f"  {ident:60s} ({count}x)")
    print()

unknown = results["unknown_params"]
if unknown:
    total_unknown = sum(sum(v.values()) for v in unknown.values())
    print(f"UNKNOWN PARAMETERS ({total_unknown} instances across {len(unknown)} actions):")
    # Show top 20 by instance count
    flat = []
    for ident, params in unknown.items():
        for param, count in params.items():
            flat.append((ident, param, count))
    flat.sort(key=lambda x: -x[2])
    for ident, param, count in flat[:20]:
        short_ident = ident.replace("is.workflow.actions.", "")
        print(f"  {short_ident:40s} {param:30s} ({count}x)")
    print()

# === Pass/Fail ===
resolution_pct = 100 * resolved / max(1, total)
param_pct = 100 * covered / max(1, total_params)

print("-" * 70)
if resolution_pct >= 95:
    print(f"Action resolution:  PASS ({resolution_pct:.1f}% >= 95%)")
else:
    print(f"Action resolution:  FAIL ({resolution_pct:.1f}% < 95%)")

if param_pct >= 90:
    print(f"Parameter coverage: PASS ({param_pct:.1f}% >= 90%)")
else:
    print(f"Parameter coverage: FAIL ({param_pct:.1f}% < 90%)")

if resolution_pct >= 95 and param_pct >= 90:
    print("\nOVERALL: PASS")
else:
    print("\nOVERALL: FAIL")
    sys.exit(1)
