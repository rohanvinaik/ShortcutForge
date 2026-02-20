#!/usr/bin/env python3
"""
Auto-generate canonical_map entries for the action catalog.

Reads action_catalog.json, derives short-form aliases for all actions,
and writes them back to _meta.canonical_map while preserving manual entries.

Strategies:
  1. Strip is.workflow.actions. prefix → short name
  2. Generate set*/toggle* aliases for X.set device actions
  3. Generate get* aliases for get.* and getX actions
  4. Strip Apple intent suffixes → readable short name
  5. Preserve all existing manual entries

Usage:
    python scripts/generate_canonical_map.py --dry-run    # Preview new entries
    python scripts/generate_canonical_map.py --apply       # Write to catalog
    python scripts/generate_canonical_map.py --stats       # Show coverage stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CATALOG_PATH = _PROJECT_ROOT / "references" / "action_catalog.json"


# ── Intent suffix stripping patterns ────────────────────────────
# Apple system intents often have long identifiers like:
#   com.apple.mobiletimer-framework.MobileTimerIntents.MTCreateAlarmIntent
# We want to extract readable names like "createalarm"

_INTENT_SUFFIX_RE = re.compile(r"Intent$", re.IGNORECASE)

# Known Apple intent class prefixes to strip
_APPLE_INTENT_PREFIXES = {
    "MT",  # MobileTimer (MTCreateAlarmIntent)
    "AX",  # Accessibility (AXSetBackgroundSoundIntent)
    "IN",  # Intents
    "REM",  # Reminders
    "SF",  # ShortcutsFramework
}


def _camel_to_lower(name: str) -> str:
    """Convert CamelCase to lowercase without separators.

    MTCreateAlarmIntent → createalarm
    AXSetBackgroundSoundIntent → setbackgroundsound
    """
    # Strip known prefixes
    for prefix in _APPLE_INTENT_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            next_char = name[len(prefix)]
            if next_char.isupper():
                name = name[len(prefix) :]
                break

    # Strip Intent suffix
    name = _INTENT_SUFFIX_RE.sub("", name)

    # CamelCase → lowercase
    return name.lower()


def _extract_intent_short_name(identifier: str) -> str | None:
    """Extract a readable short name from a system intent identifier.

    com.apple.mobiletimer-framework.MobileTimerIntents.MTCreateAlarmIntent
    → createalarm

    com.apple.AccessibilityUtilities.AXSettingsShortcuts.AXSetBackgroundSoundIntent
    → setbackgroundsound
    """
    if not identifier.startswith("com.apple."):
        return None

    # Get the last segment (class name)
    parts = identifier.split(".")
    class_name = parts[-1]

    # Must look like an intent class (CamelCase, ends with Intent or Action)
    if not (class_name.endswith("Intent") or class_name.endswith("Action")):
        return None

    short = _camel_to_lower(class_name)

    # Filter out overly generic names
    if len(short) < 4 or short in {"action", "intent", "open", "set", "get"}:
        return None

    return short


def generate_canonical_map(catalog: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Generate canonical_map entries from action catalog.

    Returns (new_entries, existing_entries):
      - new_entries: entries to add (not in existing map)
      - existing_entries: already in the map (unchanged)
    """
    actions = catalog.get("actions", {})
    existing_map = dict(catalog.get("_meta", {}).get("canonical_map", {}))

    new_entries: dict[str, str] = {}

    for identifier in actions:
        # Strategy 1: Strip is.workflow.actions. prefix
        if identifier.startswith("is.workflow.actions."):
            short = identifier[len("is.workflow.actions.") :]
            if short and short not in existing_map:
                new_entries[short] = identifier

            # Strategy 2: Device toggle aliases for X.set actions
            if short.endswith(".set"):
                feature = short[:-4]  # e.g., "bluetooth"
                # Generate set* and toggle* aliases
                set_alias = f"set{feature}"
                toggle_alias = f"toggle{feature}"
                turnon_alias = f"turnon{feature}"
                turnoff_alias = f"turnoff{feature}"

                for alias in [set_alias, toggle_alias, turnon_alias, turnoff_alias]:
                    if alias not in existing_map and alias not in new_entries:
                        new_entries[alias] = identifier

                # Also add the full-prefix variants
                set_full = f"is.workflow.actions.set{feature}"
                toggle_full = f"is.workflow.actions.toggle{feature}"
                for alias in [set_full, toggle_full]:
                    if alias not in existing_map and alias not in new_entries:
                        new_entries[alias] = identifier

        # Strategy 3: Apple system intent short names
        elif identifier.startswith("com.apple."):
            short = _extract_intent_short_name(identifier)
            if short and short not in existing_map and short not in new_entries:
                new_entries[short] = identifier

        # Strategy 4: Third-party intent short names (more conservative)
        # Only extract if the class name is clearly an intent
        else:
            parts = identifier.split(".")
            class_name = parts[-1] if parts else ""
            if class_name.endswith("Intent") and len(class_name) > 10:
                third_party_short = _camel_to_lower(class_name)
                if (
                    third_party_short
                    and len(third_party_short) >= 6
                    and third_party_short not in existing_map
                    and third_party_short not in new_entries
                ):
                    new_entries[third_party_short] = identifier

    return new_entries, existing_map


def apply_to_catalog(catalog: dict, new_entries: dict[str, str]) -> dict:
    """Apply new canonical_map entries to catalog, preserving existing."""
    if "_meta" not in catalog:
        catalog["_meta"] = {}
    if "canonical_map" not in catalog["_meta"]:
        catalog["_meta"]["canonical_map"] = {}

    merged = dict(catalog["_meta"]["canonical_map"])
    merged.update(new_entries)

    # Sort for readability
    catalog["_meta"]["canonical_map"] = dict(sorted(merged.items()))
    return catalog


def show_stats(catalog: dict, new_entries: dict, existing: dict):
    """Print coverage statistics."""
    actions = catalog.get("actions", {})
    total = len(actions)

    # Count by prefix
    workflow_count = sum(1 for a in actions if a.startswith("is.workflow.actions."))
    apple_count = sum(1 for a in actions if a.startswith("com.apple."))
    third_party_count = total - workflow_count - apple_count

    # Coverage
    all_map = {**existing, **new_entries}
    covered = set()
    for short, full in all_map.items():
        covered.add(full)
    coverage = len(covered.intersection(actions)) / total * 100 if total else 0

    print("\nCanonical Map Coverage Statistics")
    print(f"{'=' * 50}")
    print(f"Total actions:       {total}")
    print(f"  is.workflow.*:     {workflow_count}")
    print(f"  com.apple.*:       {apple_count}")
    print(f"  Third-party:       {third_party_count}")
    print(f"\nExisting map:        {len(existing)} entries")
    print(f"New entries:         {len(new_entries)} entries")
    print(f"Total after merge:   {len(existing) + len(new_entries)} entries")
    print(
        f"Action coverage:     {len(covered.intersection(actions))}/{total} ({coverage:.1f}%)"
    )

    # Breakdown new entries by type
    new_workflow = sum(
        1 for v in new_entries.values() if v.startswith("is.workflow.actions.")
    )
    new_apple = sum(1 for v in new_entries.values() if v.startswith("com.apple."))
    new_third = len(new_entries) - new_workflow - new_apple
    print("\nNew entry breakdown:")
    print(f"  is.workflow.*:     {new_workflow}")
    print(f"  com.apple.*:       {new_apple}")
    print(f"  Third-party:       {new_third}")

    # Uncovered actions
    uncovered = set(actions.keys()) - covered
    if uncovered:
        print(f"\nUncovered actions ({len(uncovered)}):")
        for a in sorted(uncovered)[:20]:
            print(f"  {a}")
        if len(uncovered) > 20:
            print(f"  ... and {len(uncovered) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate canonical_map entries for action catalog",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview new entries without writing",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write new entries to action_catalog.json",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show coverage statistics",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(_CATALOG_PATH),
        help="Path to action_catalog.json",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.apply and not args.stats:
        parser.print_help()
        sys.exit(1)

    with open(args.catalog) as f:
        catalog = json.load(f)

    new_entries, existing = generate_canonical_map(catalog)

    if args.stats or args.dry_run:
        show_stats(catalog, new_entries, existing)

    if args.dry_run:
        print(f"\nNew entries to add ({len(new_entries)}):")
        for short, full in sorted(new_entries.items()):
            print(f"  {short:40s} → {full}")

    if args.apply:
        catalog = apply_to_catalog(catalog, new_entries)
        with open(args.catalog, "w") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nWritten {len(new_entries)} new entries to {args.catalog}")
        print(f"Total canonical_map size: {len(catalog['_meta']['canonical_map'])}")


if __name__ == "__main__":
    main()
