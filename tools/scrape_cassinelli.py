"""
Scrape Apple Shortcuts actions from Matthew Cassinelli's WordPress API.

Outputs:
- raw JSON API dump
- cleaned action records used by downstream scripts
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://matthewcassinelli.com/wp-json/wp/v2/actions"
PER_PAGE = 100

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_RAW = BASE_DIR / "references" / "cassinelli_actions_raw.json"
DEFAULT_OUTPUT_CLEAN = BASE_DIR / "references" / "cassinelli_actions_clean.json"


def fetch_page(page_num: int) -> tuple[list, int | None]:
    """Fetch a single page of actions from the API."""
    url = f"{BASE_URL}?per_page={PER_PAGE}&page={page_num}"
    print(f"  Fetching page {page_num}: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "ShortcutsCompiler/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            total = resp.headers.get("X-WP-Total", "?")
            total_pages = resp.headers.get("X-WP-TotalPages", "?")
            print(f"    Got {len(data)} actions (total: {total}, pages: {total_pages})")
            return data, int(total_pages) if total_pages != "?" else None
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print(f"    Page {page_num} returned 400 - likely past last page")
            return [], 0
        raise


def fetch_all_actions() -> list:
    """Paginate through all actions."""
    all_actions = []
    page = 1
    known_total_pages = None

    while True:
        data, total_pages = fetch_page(page)
        if total_pages is not None:
            known_total_pages = total_pages

        if not data:
            break

        all_actions.extend(data)

        if known_total_pages and page >= known_total_pages:
            break

        page += 1
        time.sleep(0.5)

    return all_actions


def clean_action(raw: dict) -> dict:
    """Extract the fields we care about from a raw API response."""
    acf = raw.get("acf", {}) or {}

    return {
        "wp_id": raw.get("id"),
        "slug": raw.get("slug", ""),
        "url": raw.get("link", ""),
        "title": raw.get("title", {}).get("rendered", ""),
        "name": acf.get("name", ""),
        "description": acf.get("description", ""),
        "identifier": acf.get("identifier", ""),
        "parameters": acf.get("parameters", ""),
        "input": acf.get("input", ""),
        "result": acf.get("result", ""),
        "score": acf.get("score", ""),
        "notes": acf.get("notes", ""),
        "emoji": acf.get("emoji", ""),
        "apps_text": acf.get("apps_text", ""),
        "works_well_with_text": acf.get("works_well_with_text", ""),
        "related_shortcuts_text": acf.get("related_shortcuts_text", ""),
        "automations_text": acf.get("automations_text", ""),
        "platform": acf.get("platform", ""),
        "action_type_ids": raw.get("action_type", []),
        "action_group_ids": raw.get("action_groups", []),
        "stage_ids": raw.get("stages", []),
        "prompt_ids": raw.get("prompt", []),
        "category_ids": raw.get("categories", []),
        "tag_ids": raw.get("tags", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Cassinelli action catalog")
    parser.add_argument("--output-raw", type=Path, default=DEFAULT_OUTPUT_RAW)
    parser.add_argument("--output-clean", type=Path, default=DEFAULT_OUTPUT_CLEAN)
    args = parser.parse_args()

    print("=" * 60)
    print("Fetching all actions from Cassinelli WordPress API")
    print("=" * 60)

    raw_actions = fetch_all_actions()
    print(f"\nTotal actions fetched: {len(raw_actions)}")

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    with args.output_raw.open("w") as f:
        json.dump(raw_actions, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to {args.output_raw}")

    clean_actions = [clean_action(a) for a in raw_actions]
    clean_actions.sort(key=lambda a: a.get("identifier", "") or "zzz")

    with_identifier = [a for a in clean_actions if a.get("identifier")]
    with_params = [a for a in clean_actions if a.get("parameters")]
    with_description = [a for a in clean_actions if a.get("description")]

    print(f"\nCleaned actions: {len(clean_actions)}")
    print(f"  With identifier: {len(with_identifier)}")
    print(f"  With parameters: {len(with_params)}")
    print(f"  With description: {len(with_description)}")

    args.output_clean.parent.mkdir(parents=True, exist_ok=True)
    with args.output_clean.open("w") as f:
        json.dump(clean_actions, f, indent=2, ensure_ascii=False)
    print(f"Clean data saved to {args.output_clean}")

    print("\n" + "=" * 60)
    print("Sample actions (first 10 with identifiers):")
    print("=" * 60)
    for a in with_identifier[:10]:
        print(f"  {a['identifier']}")
        print(f"    Name: {a['name']}")
        params_text = a.get("parameters", "")
        if len(params_text) > 80:
            print(f"    Params: {params_text[:80]}...")
        else:
            print(f"    Params: {params_text}")
        print(f"    Input: {a['input']}")
        print(f"    Result: {a['result']}")
        print()


if __name__ == "__main__":
    main()
