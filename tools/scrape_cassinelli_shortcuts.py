"""
Scrape shortcuts from Cassinelli's WordPress API.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://matthewcassinelli.com/wp-json/wp/v2/shortcuts"
PER_PAGE = 100

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_RAW = BASE_DIR / "references" / "cassinelli_shortcuts_raw.json"
DEFAULT_OUTPUT_CLEAN = BASE_DIR / "references" / "cassinelli_shortcuts_library.json"


def fetch_page(page_num: int) -> tuple[list, int | None]:
    url = f"{BASE_URL}?per_page={PER_PAGE}&page={page_num}"
    print(f"  Fetching page {page_num}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ShortcutsCompiler/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            total = resp.headers.get("X-WP-Total", "?")
            total_pages = resp.headers.get("X-WP-TotalPages", "?")
            print(
                f"    Got {len(data)} shortcuts (total: {total}, pages: {total_pages})"
            )
            return data, int(total_pages) if total_pages != "?" else None
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print(f"    Page {page_num} returned 400 - past last page")
            return [], 0
        raise


def fetch_all() -> list:
    all_shortcuts = []
    page = 1
    known_total_pages = None
    while True:
        data, total_pages = fetch_page(page)
        if total_pages is not None:
            known_total_pages = total_pages
        if not data:
            break
        all_shortcuts.extend(data)
        if known_total_pages and page >= known_total_pages:
            break
        page += 1
        time.sleep(0.5)
    return all_shortcuts


def clean_shortcut(raw: dict) -> dict:
    acf = raw.get("acf", {}) or {}
    return {
        "wp_id": raw.get("id"),
        "slug": raw.get("slug", ""),
        "url": raw.get("link", ""),
        "title": raw.get("title", {}).get("rendered", ""),
        "name": acf.get("shortcut_name", ""),
        "description": acf.get("description", ""),
        "action_count": acf.get("action_count", ""),
        "icloud_link": acf.get("icloud_link", ""),
        "actions": acf.get("actions", []),
        "open_shortcut_link": acf.get("open_shortcut_link", ""),
        "run_shortcut_link": acf.get("run_shortcut_link", ""),
        "hex_color": acf.get("hex_color", ""),
        "membership": raw.get("membership", []),
        "folder_ids": raw.get("folders", []),
        "category_ids": raw.get("categories", []),
        "tag_ids": raw.get("tags", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Cassinelli shortcuts library")
    parser.add_argument("--output-raw", type=Path, default=DEFAULT_OUTPUT_RAW)
    parser.add_argument("--output-clean", type=Path, default=DEFAULT_OUTPUT_CLEAN)
    args = parser.parse_args()

    print("=" * 60)
    print("Fetching all shortcuts from Cassinelli WordPress API")
    print("=" * 60)

    raw = fetch_all()
    print(f"\nTotal shortcuts fetched: {len(raw)}")

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    with args.output_raw.open("w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to {args.output_raw}")

    clean = [clean_shortcut(s) for s in raw]
    clean.sort(key=lambda s: s.get("title", ""))

    with_icloud = [s for s in clean if s.get("icloud_link")]
    with_actions = [s for s in clean if s.get("actions")]
    free = [
        s
        for s in clean
        if not s.get("membership") or 1805 not in s.get("membership", [])
    ]

    print(f"\nCleaned shortcuts: {len(clean)}")
    print(f"  With iCloud link: {len(with_icloud)}")
    print(f"  With action list: {len(with_actions)}")
    print(f"  Free (no membership gate): {len(free)}")

    counts = [
        int(s["action_count"])
        for s in clean
        if s.get("action_count") and str(s["action_count"]).isdigit()
    ]
    if counts:
        print("\nAction counts:")
        print(
            f"  Min: {min(counts)}, Max: {max(counts)}, Avg: {sum(counts) / len(counts):.0f}"
        )
        simple = sum(1 for c in counts if c <= 5)
        medium = sum(1 for c in counts if 5 < c <= 20)
        complex_ = sum(1 for c in counts if 20 < c <= 50)
        mega = sum(1 for c in counts if c > 50)
        print(f"  Simple (1-5): {simple}")
        print(f"  Medium (6-20): {medium}")
        print(f"  Complex (21-50): {complex_}")
        print(f"  Mega (50+): {mega}")

    args.output_clean.parent.mkdir(parents=True, exist_ok=True)
    with args.output_clean.open("w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\nClean data saved to {args.output_clean}")

    with_links_and_counts = [
        s
        for s in clean
        if s.get("icloud_link")
        and s.get("action_count")
        and str(s["action_count"]).isdigit()
    ]
    with_links_and_counts.sort(key=lambda s: int(s["action_count"]), reverse=True)

    print(f"\n{'=' * 60}")
    print("Top 15 most complex shortcuts with iCloud links:")
    print("=" * 60)
    for s in with_links_and_counts[:15]:
        print(f"  [{s['action_count']:>3s} actions] {s['title']}")
        print(f"    iCloud: {s['icloud_link'][:80]}...")
        desc = s.get("description", "")
        if len(desc) > 100:
            print(f"    Desc: {desc[:100]}...")
        else:
            print(f"    Desc: {desc}")
        print()


if __name__ == "__main__":
    main()
