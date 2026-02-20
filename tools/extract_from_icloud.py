"""
Apple Shortcuts iCloud Extractor
=================================
Given iCloud shortcut links, downloads the .shortcut files and extracts
every unique action type with its full parameter structure.

This is the bridge between "we know what actions exist" (catalog) and
"we know exactly how to render them" (compiler methods).

Usage:
    python extract_from_icloud.py <icloud_url> [<icloud_url2> ...]
    python extract_from_icloud.py --from-library <N>  # Extract from top N complex shortcuts in library

Output:
    - Downloaded .shortcut files in ./downloaded/
    - extracted_actions.json — every unique action type with parameter structures
    - extraction_report.txt — human-readable summary
"""

import json
import os
import plistlib
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

# =============================================================================
# CONFIGURATION
# =============================================================================

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloaded")
EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "..", "references", "extracted")
LIBRARY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "references", "cassinelli_shortcuts_library.json"
)


# =============================================================================
# iCLOUD API
# =============================================================================


def icloud_link_to_api_url(icloud_url: str) -> str:
    """
    Convert an iCloud shortcut link to the API endpoint that returns metadata
    including the download URL.

    Input:  https://www.icloud.com/shortcuts/86cd1eeabddc44188607238acd4cc7ef
    Output: https://www.icloud.com/shortcuts/api/records/86cd1eeabddc44188607238acd4cc7ef
    """
    # Extract the UUID from various URL formats
    match = re.search(r"/shortcuts/([a-f0-9-]+)", icloud_url)
    if not match:
        raise ValueError(f"Could not extract shortcut ID from URL: {icloud_url}")
    shortcut_id = match.group(1)
    return f"https://www.icloud.com/shortcuts/api/records/{shortcut_id}"


def get_download_url(icloud_url: str) -> tuple[str, str]:
    """
    Fetch the API record and extract the download URL and shortcut name.
    Prefers the UNSIGNED shortcut (plain plist) over the signed version (AEA archive).
    Returns (download_url, name).
    """
    api_url = icloud_link_to_api_url(icloud_url)
    req = urllib.request.Request(
        api_url, headers={"User-Agent": "ShortcutsCompiler/1.0"}
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    fields = data.get("fields", {})
    name = fields.get("name", {}).get("value", "Unknown")

    # Prefer unsigned shortcut (plain plist, parseable without macOS tools)
    # Fall back to signed version if unsigned isn't available
    unsigned = fields.get("shortcut", {}).get("value", {})
    download_url = unsigned.get("downloadURL", "")

    if not download_url:
        signed = fields.get("signedShortcut", {}).get("value", {})
        download_url = signed.get("downloadURL", "")
        if download_url:
            print(
                "    WARNING: Only signed version available (AEA format, needs macOS to parse)"
            )

    if not download_url:
        raise ValueError(f"No downloadURL found in API response for {icloud_url}")

    return download_url, name


def download_shortcut(
    icloud_url: str, output_dir: str = DOWNLOAD_DIR
) -> tuple[str, str]:
    """
    Download a .shortcut file from an iCloud link.
    Returns (filepath, name).
    """
    os.makedirs(output_dir, exist_ok=True)

    download_url, name = get_download_url(icloud_url)

    # Sanitize filename
    safe_name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    filepath = os.path.join(output_dir, f"{safe_name}.shortcut")

    req = urllib.request.Request(
        download_url, headers={"User-Agent": "ShortcutsCompiler/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    with open(filepath, "wb") as f:
        f.write(data)

    return filepath, name


# =============================================================================
# PLIST EXTRACTION
# =============================================================================


def load_shortcut(filepath: str) -> dict:
    """Load a .shortcut plist file."""
    with open(filepath, "rb") as f:
        return plistlib.load(f)


def extract_actions(plist_data: dict) -> list[dict]:
    """Extract the action list from a shortcut plist."""
    return plist_data.get("WFWorkflowActions", [])


def anonymize_uuids(obj: Any) -> Any:
    """
    Replace UUIDs with placeholder strings so we can see the structure
    without the noise of specific UUIDs.
    """
    uuid_pattern = re.compile(
        r"^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$", re.IGNORECASE
    )

    if isinstance(obj, str):
        if uuid_pattern.match(obj):
            return "<UUID>"
        return obj
    elif isinstance(obj, dict):
        return {k: anonymize_uuids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [anonymize_uuids(item) for item in obj]
    elif isinstance(obj, bytes):
        return "<BYTES>"
    return obj


def extract_unique_actions(actions: list[dict]) -> dict:
    """
    Group actions by identifier and collect unique parameter structures.
    Returns {identifier: {"count": N, "parameter_keys": [...], "examples": [...]}}
    """
    by_type = defaultdict(
        lambda: {"count": 0, "parameter_keys_seen": set(), "examples": []}
    )

    for action in actions:
        ident = action.get("WFWorkflowActionIdentifier", "unknown")
        params = action.get("WFWorkflowActionParameters", {})
        entry = by_type[ident]
        entry["count"] += 1

        # Track all parameter keys we've seen for this action type
        for key in params.keys():
            entry["parameter_keys_seen"].add(key)

        # Keep up to 3 unique examples (by parameter key set)
        key_set = frozenset(params.keys())
        existing_key_sets = [frozenset(ex.keys()) for ex in entry["examples"]]
        if key_set not in existing_key_sets and len(entry["examples"]) < 3:
            entry["examples"].append(anonymize_uuids(params))

    # Convert sets to sorted lists for JSON serialization
    result = {}
    for ident, data in sorted(by_type.items()):
        result[ident] = {
            "count": data["count"],
            "parameter_keys": sorted(data["parameter_keys_seen"]),
            "examples": data["examples"],
        }

    return result


def extract_from_file(filepath: str) -> dict:
    """
    Full extraction from a single .shortcut file.
    Returns structured analysis.
    """
    plist_data = load_shortcut(filepath)
    actions = extract_actions(plist_data)

    # Envelope info
    envelope = {
        "client_release": plist_data.get("WFWorkflowClientRelease", ""),
        "client_version": plist_data.get("WFWorkflowClientVersion", ""),
        "min_version": plist_data.get("WFWorkflowMinimumClientVersion", ""),
        "input_types": plist_data.get("WFWorkflowInputContentItemClasses", []),
        "workflow_types": plist_data.get("WFWorkflowTypes", []),
        "icon": plist_data.get("WFWorkflowIcon", {}),
    }

    unique_actions = extract_unique_actions(actions)

    # Find control flow patterns
    control_flow = []
    for i, action in enumerate(actions):
        ident = action.get("WFWorkflowActionIdentifier", "")
        params = action.get("WFWorkflowActionParameters", {})
        if "WFControlFlowMode" in params:
            control_flow.append(
                {
                    "index": i,
                    "identifier": ident,
                    "mode": params["WFControlFlowMode"],
                    "has_grouping_id": "GroupingIdentifier" in params,
                }
            )

    return {
        "total_actions": len(actions),
        "unique_action_types": len(unique_actions),
        "envelope": envelope,
        "actions": unique_actions,
        "control_flow_summary": control_flow,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def merge_extractions(extractions: list[dict]) -> dict:
    """
    Merge multiple shortcut extractions into a single comprehensive action reference.
    Keeps the richest examples for each action type.
    """
    merged = {}

    for extraction in extractions:
        for ident, data in extraction.get("actions", {}).items():
            if ident not in merged:
                merged[ident] = {
                    "total_occurrences": 0,
                    "shortcuts_seen_in": 0,
                    "parameter_keys": set(),
                    "examples": [],
                }

            entry = merged[ident]
            entry["total_occurrences"] += data["count"]
            entry["shortcuts_seen_in"] += 1

            for key in data.get("parameter_keys", []):
                entry["parameter_keys"].add(key)

            # Add unique examples (by key set, up to 5 total)
            for example in data.get("examples", []):
                key_set = frozenset(example.keys())
                existing = [frozenset(ex.keys()) for ex in entry["examples"]]
                if key_set not in existing and len(entry["examples"]) < 5:
                    entry["examples"].append(example)

    # Convert sets for JSON
    for ident, data in merged.items():
        data["parameter_keys"] = sorted(data["parameter_keys"])

    return dict(sorted(merged.items()))


def generate_report(merged: dict, shortcuts_processed: list[str]) -> str:
    """Generate a human-readable extraction report."""
    lines = [
        "=" * 70,
        "APPLE SHORTCUTS ACTION EXTRACTION REPORT",
        "=" * 70,
        f"Shortcuts processed: {len(shortcuts_processed)}",
        f"Unique action types found: {len(merged)}",
        "",
    ]

    for name in shortcuts_processed:
        lines.append(f"  - {name}")
    lines.append("")

    # Group by namespace
    namespaces = defaultdict(list)
    for ident in merged:
        if ident.startswith("is.workflow.actions."):
            ns = "core"
        elif ident.startswith("com.apple."):
            parts = ident.split(".")
            ns = ".".join(parts[:3])
        else:
            ns = "other"
        namespaces[ns].append(ident)

    for ns in sorted(namespaces):
        idents = namespaces[ns]
        lines.append(f"\n--- {ns} ({len(idents)} actions) ---")
        for ident in sorted(idents):
            data = merged[ident]
            short = ident.replace("is.workflow.actions.", "") if ns == "core" else ident
            lines.append(
                f"  {short:<45s} "
                f"seen {data['total_occurrences']:>3d}x in {data['shortcuts_seen_in']} shortcuts  "
                f"params: {data['parameter_keys']}"
            )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    icloud_urls = []
    shortcuts_names = []

    if len(sys.argv) > 1 and sys.argv[1] == "--from-library":
        # Load from our scraped library
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"Loading top {n} most complex shortcuts from library...")

        with open(LIBRARY_FILE) as f:
            library = json.load(f)

        # Filter to those with iCloud links and action counts, sort by complexity
        with_links = [
            s
            for s in library
            if s.get("icloud_link")
            and s.get("action_count")
            and str(s["action_count"]).isdigit()
            and int(s["action_count"]) > 0
        ]
        with_links.sort(key=lambda s: int(s["action_count"]), reverse=True)

        for s in with_links[:n]:
            icloud_urls.append(s["icloud_link"])
            shortcuts_names.append(f"{s['title']} ({s['action_count']} actions)")

    elif len(sys.argv) > 1:
        icloud_urls = sys.argv[1:]
    else:
        print("Usage:")
        print("  python extract_from_icloud.py <icloud_url> [<icloud_url2> ...]")
        print("  python extract_from_icloud.py --from-library <N>")
        print("")
        print("Examples:")
        print(
            "  python extract_from_icloud.py https://www.icloud.com/shortcuts/86cd1eea..."
        )
        print("  python extract_from_icloud.py --from-library 15")
        sys.exit(1)

    print(f"\nProcessing {len(icloud_urls)} shortcuts...")
    all_extractions = []
    processed_names = []

    for i, url in enumerate(icloud_urls):
        label = shortcuts_names[i] if i < len(shortcuts_names) else url[:60]
        print(f"\n[{i + 1}/{len(icloud_urls)}] {label}")

        try:
            # Download
            print("  Downloading...")
            filepath, name = download_shortcut(url)
            print(f"  Saved: {filepath}")
            processed_names.append(name)

            # Extract
            print("  Extracting actions...")
            extraction = extract_from_file(filepath)
            extraction["source_name"] = name
            extraction["source_url"] = url
            all_extractions.append(extraction)

            print(
                f"  Found {extraction['total_actions']} actions, "
                f"{extraction['unique_action_types']} unique types"
            )

            time.sleep(0.5)  # Be polite to Apple's servers

        except Exception as e:
            print(f"  ERROR: {e}")
            processed_names.append(f"FAILED: {url[:60]}")
            continue

    if not all_extractions:
        print("\nNo shortcuts were successfully processed.")
        sys.exit(1)

    # Merge all extractions
    print(f"\n{'=' * 60}")
    print("Merging extractions...")
    merged = merge_extractions(all_extractions)
    print(f"Total unique action types across all shortcuts: {len(merged)}")

    # Save individual extractions
    individual_path = os.path.join(EXTRACTED_DIR, "individual_extractions.json")
    with open(individual_path, "w") as f:
        json.dump(all_extractions, f, indent=2, ensure_ascii=False, default=str)
    print(f"Individual extractions: {individual_path}")

    # Save merged action reference
    merged_path = os.path.join(EXTRACTED_DIR, "extracted_actions.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False, default=str)
    print(f"Merged action reference: {merged_path}")

    # Generate report
    report = generate_report(merged, processed_names)
    report_path = os.path.join(EXTRACTED_DIR, "extraction_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report: {report_path}")

    # Print summary
    print(f"\n{report}")


if __name__ == "__main__":
    main()
