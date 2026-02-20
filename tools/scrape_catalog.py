#!/usr/bin/env python3
"""
Catalog Scraper: Auto-regenerate action_catalog.json from macOS system files.

Apple adds new Shortcuts actions every macOS/iOS release. This script scrapes
the local system to discover all available actions, their parameters, and
metadata — so the catalog stays fresh without manual maintenance.

Sources scraped:
  1. ActionKit.framework/Resources/Base.lproj/Actions.intentdefinition
     → Classic WF* intents (SplitText, AskForInput, SetWiFi, etc.)
  2. ActionKit.framework/Resources/Metadata.appintents/extract.actionsdata
     → Newer AppIntents-era actions
  3. All extract.actionsdata files across /System/Library/
     → Per-framework actions (Finder, Reminders, Calendar, HomeKit, etc.)
  4. Downloaded .shortcut corpus (if available)
     → Observed parameters from real-world usage

Usage:
    python3 scrape_catalog.py [--merge-corpus] [--output path/to/catalog.json]

The --merge-corpus flag enriches the scraped catalog with observed parameter
data from the downloaded shortcuts corpus (if present in ../downloaded/).
Without it, only system-level metadata is included.
"""

from __future__ import annotations

import json
import os
import plistlib
from pathlib import Path

# System paths for action definitions
ACTIONKIT_FRAMEWORK = Path(
    "/System/Library/PrivateFrameworks/ActionKit.framework/Versions/A/Resources"
)
INTENTS_FILE = ACTIONKIT_FRAMEWORK / "Base.lproj" / "Actions.intentdefinition"
APPINTENTS_FILE = ACTIONKIT_FRAMEWORK / "Metadata.appintents" / "extract.actionsdata"
SYSTEM_SEARCH_ROOT = Path("/System/Library")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "references" / "action_catalog.json"
DOWNLOADS_DIR = BASE_DIR / "downloaded"


def scrape_intentdefinition(path: Path) -> dict[str, dict]:
    """Scrape Actions.intentdefinition for classic WF* intents."""
    actions = {}
    if not path.exists():
        print(f"  [skip] {path} not found")
        return actions

    with path.open("rb") as f:
        plist = plistlib.load(f)

    intents = plist.get("INIntents", [])
    for intent in intents:
        class_name = intent.get("INIntentClassPrefix", "") + intent.get(
            "INIntentName", ""
        )
        identifier = intent.get("INIntentManagedPropertyList", {}).get(
            "WFWorkflowActionIdentifier", ""
        )
        if not identifier:
            # Try constructing from class name
            identifier = (
                f"is.workflow.actions.{class_name.lower()}" if class_name else ""
            )
        if not identifier:
            continue

        # Extract parameters
        params = {}
        for param in intent.get("INIntentParameters", []):
            param_name = param.get("INIntentParameterName", "")
            param_type = param.get("INIntentParameterType", "")
            if param_name:
                params[param_name] = {
                    "type": param_type,
                    "source": "intentdefinition",
                }

        description = intent.get("INIntentDescriptionID", "") or intent.get(
            "INIntentDescription", ""
        )
        title = intent.get("INIntentTitle", "") or class_name

        actions[identifier] = {
            "identifier": identifier,
            "name": title,
            "description": description,
            "source": "ActionKit.intentdefinition",
            "observed_params": params,
        }

    print(f"  [ok] {path.name}: {len(actions)} intents")
    return actions


def scrape_actionsdata(path: Path, source_label: str = "") -> dict[str, dict]:
    """Scrape an extract.actionsdata JSON file for AppIntents-era actions."""
    actions = {}
    if not path.exists():
        print(f"  [skip] {path} not found")
        return actions

    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return actions

    # extract.actionsdata is typically a list of action definitions
    action_list = data if isinstance(data, list) else data.get("actions", [])
    if not isinstance(action_list, list):
        return actions

    for action_def in action_list:
        if not isinstance(action_def, dict):
            continue

        identifier = action_def.get("identifier", "") or action_def.get("id", "")
        if not identifier:
            continue

        # Extract parameters from the schema
        params = {}
        for param in action_def.get("parameters", []):
            if isinstance(param, dict):
                pname = param.get("name", "") or param.get("key", "")
                if pname:
                    params[pname] = {
                        "type": param.get("type", ""),
                        "source": "actionsdata",
                    }

        title = action_def.get("title", {})
        if isinstance(title, dict):
            title = title.get("key", "") or title.get("value", "")
        description = action_def.get("descriptionMetadata", {})
        if isinstance(description, dict):
            description = description.get("descriptionText", {})
            if isinstance(description, dict):
                description = description.get("key", "")

        label = source_label or path.parent.parent.name
        actions[identifier] = {
            "identifier": identifier,
            "name": str(title),
            "description": str(description),
            "source": label,
            "observed_params": params,
        }

    if actions:
        print(f"  [ok] {label}: {len(actions)} actions")
    return actions


def scrape_system_actionsdata() -> dict[str, dict]:
    """Find and scrape all extract.actionsdata files across the system."""
    all_actions = {}

    print("\nScanning system frameworks for extract.actionsdata...")
    count = 0
    for root, dirs, files in os.walk(SYSTEM_SEARCH_ROOT):
        if "extract.actionsdata" in files:
            path = Path(root) / "extract.actionsdata"
            # Derive a human-readable source label
            # e.g., /System/Library/.../Finder.app/... → Finder.app
            parts = path.parts
            source = "system"
            for part in parts:
                if part.endswith((".app", ".framework", ".appex")):
                    source = part
                    break
            actions = scrape_actionsdata(path, source)
            all_actions.update(actions)
            count += 1

    print(
        f"  Scanned {count} actionsdata files, found {len(all_actions)} total actions"
    )
    return all_actions


def scrape_corpus(
    downloads_dir: Path, existing_actions: dict[str, dict]
) -> dict[str, dict]:
    """Enrich the catalog with observed parameters from downloaded shortcuts."""
    if not downloads_dir.exists():
        print(f"\n  [skip] Corpus not found at {downloads_dir}")
        return existing_actions

    shortcut_files = list(downloads_dir.glob("*.shortcut"))
    print(f"\nEnriching catalog from corpus ({len(shortcut_files)} shortcuts)...")

    param_counts: dict[str, dict[str, int]] = {}  # identifier → {param_name: count}

    for filepath in shortcut_files:
        try:
            with filepath.open("rb") as f:
                plist = plistlib.load(f)
        except Exception:
            continue

        for action in plist.get("WFWorkflowActions", []):
            identifier = action.get("WFWorkflowActionIdentifier", "")
            params = action.get("WFWorkflowActionParameters", {})

            if identifier not in param_counts:
                param_counts[identifier] = {}

            for pname in params:
                if pname in (
                    "UUID",
                    "GroupingIdentifier",
                    "WFControlFlowMode",
                    "CustomOutputName",
                ):
                    continue
                param_counts[identifier][pname] = (
                    param_counts[identifier].get(pname, 0) + 1
                )

    # Merge observed params into existing actions
    for identifier, pcounts in param_counts.items():
        if identifier not in existing_actions:
            # New action discovered from corpus
            short = identifier
            if short.startswith("is.workflow.actions."):
                short = short[len("is.workflow.actions.") :]
            existing_actions[identifier] = {
                "identifier": identifier,
                "name": short,
                "description": "",
                "source": "corpus",
                "observed_params": {},
            }

        action = existing_actions[identifier]
        for pname, count in pcounts.items():
            if pname not in action["observed_params"]:
                action["observed_params"][pname] = {"count": count, "source": "corpus"}
            else:
                action["observed_params"][pname]["count"] = count

    print(f"  Enriched {len(param_counts)} actions from corpus")
    return existing_actions


def build_canonical_map(actions: dict[str, dict]) -> dict[str, str]:
    """Build the canonical_map: short_name → full_identifier."""
    cmap = {}
    for identifier in actions:
        if identifier.startswith("is.workflow.actions."):
            short = identifier[len("is.workflow.actions.") :]
            if short not in cmap or len(identifier) < len(cmap.get(short, "")):
                cmap[short] = identifier
    return cmap


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape macOS for Shortcuts action catalog"
    )
    parser.add_argument(
        "--merge-corpus",
        action="store_true",
        help="Enrich catalog with observed params from downloaded shortcuts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    all_actions: dict[str, dict] = {}

    # 1. Scrape ActionKit intentdefinition
    print("Scraping ActionKit.framework...")
    intents = scrape_intentdefinition(INTENTS_FILE)
    all_actions.update(intents)

    # 2. Scrape ActionKit actionsdata
    appintents = scrape_actionsdata(APPINTENTS_FILE, "ActionKit.appintents")
    all_actions.update(appintents)

    # 3. Scrape all system actionsdata files
    system = scrape_system_actionsdata()
    all_actions.update(system)

    # 4. Optionally enrich from corpus
    if args.merge_corpus:
        all_actions = scrape_corpus(DOWNLOADS_DIR, all_actions)

    # 5. Build canonical map
    cmap = build_canonical_map(all_actions)

    # 6. Write catalog
    catalog = {
        "_meta": {
            "total_actions": len(all_actions),
            "canonical_map": cmap,
            "sources": sorted(set(a.get("source", "") for a in all_actions.values())),
            "generated_by": "scrape_catalog.py",
        },
        "actions": all_actions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nCatalog written to {args.output}")
    print(f"  Total actions: {len(all_actions)}")
    print(f"  Canonical map: {len(cmap)} short names")
    print(f"  Sources: {', '.join(catalog['_meta']['sources'])}")


if __name__ == "__main__":
    main()
