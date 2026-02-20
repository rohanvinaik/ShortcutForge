#!/usr/bin/env python3
"""
Bulk downloader for all remaining Cassinelli shortcuts.
Reads the library JSON, skips already-downloaded entries, downloads the rest.
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloaded")
LIBRARY_FILE = os.path.join(BASE_DIR, "references", "cassinelli_shortcuts_library.json")
FAILURE_LOG = os.path.join(BASE_DIR, "scripts", "download_failures.json")


def sanitize_filename(name):
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def normalize(name):
    return re.sub(r"[\s_]+", "", name).lower()


def download_one(icloud_url, output_dir):
    """Download a single shortcut from iCloud. Returns (filepath, name, format_note)."""
    match = re.search(r"/shortcuts/([a-f0-9-]+)", icloud_url)
    if not match:
        raise ValueError("Bad iCloud URL: " + icloud_url)
    api_url = "https://www.icloud.com/shortcuts/api/records/" + match.group(1)

    req = urllib.request.Request(
        api_url, headers={"User-Agent": "ShortcutsCompiler/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    fields = data.get("fields", {})
    name = fields.get("name", {}).get("value", "Unknown")

    unsigned = fields.get("shortcut", {}).get("value", {})
    dl_url = unsigned.get("downloadURL", "")
    fmt = "unsigned"

    if not dl_url:
        signed = fields.get("signedShortcut", {}).get("value", {})
        dl_url = signed.get("downloadURL", "")
        fmt = "signed_only"

    if not dl_url:
        raise ValueError("No downloadURL in API response")

    safe = sanitize_filename(name) or sanitize_filename(icloud_url.split("/")[-1])
    filepath = os.path.join(output_dir, safe + ".shortcut")

    req2 = urllib.request.Request(
        dl_url, headers={"User-Agent": "ShortcutsCompiler/1.0"}
    )
    with urllib.request.urlopen(req2, timeout=60) as resp2:
        blob = resp2.read()

    with open(filepath, "wb") as f:
        f.write(blob)

    return filepath, name, fmt


def main():
    t0 = datetime.now()
    print("=== Cassinelli Shortcuts Bulk Downloader ===")
    print("Started: " + t0.strftime("%Y-%m-%d %H:%M:%S"))
    print()

    with open(LIBRARY_FILE) as f:
        library = json.load(f)
    print("Library entries: " + str(len(library)))

    with_links = [e for e in library if e.get("icloud_link")]
    print("With iCloud links: " + str(len(with_links)))

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    existing = set()
    for fn in os.listdir(DOWNLOAD_DIR):
        if fn.endswith(".shortcut"):
            existing.add(normalize(fn[: -len(".shortcut")]))
    print("Already downloaded: " + str(len(existing)))

    to_dl = []
    skipped = 0
    for entry in with_links:
        name = entry.get("name") or entry.get("title") or ""
        san = sanitize_filename(name)
        if normalize(san) in existing or normalize(name) in existing:
            skipped += 1
        else:
            to_dl.append(entry)

    print("Skipped (already have): " + str(skipped))
    print("To download: " + str(len(to_dl)))
    print()

    if not to_dl:
        print("Nothing to download!")
        return

    ok = 0
    signed_ct = 0
    fails = []

    for i, entry in enumerate(to_dl):
        name = entry.get("name") or entry.get("title") or "Unknown"
        link = entry["icloud_link"]

        if i > 0 and i % 50 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            rate = ok / elapsed * 3600 if elapsed > 0 else 0
            msg = "--- %d/%d | %d OK | %d fail | %ds | ~%d/hr ---" % (
                i,
                len(to_dl),
                ok,
                len(fails),
                elapsed,
                rate,
            )
            print("\n" + msg + "\n")

        try:
            fp, dl_name, fmt = download_one(link, DOWNLOAD_DIR)
            ok += 1
            if fmt == "signed_only":
                signed_ct += 1
            base = os.path.basename(fp)
            if base.endswith(".shortcut"):
                existing.add(normalize(base[: -len(".shortcut")]))
            if i < 10 or ok % 100 == 0:
                print("  [%d/%d] OK: %s (%s)" % (i + 1, len(to_dl), name, fmt))

        except urllib.error.HTTPError as e:
            reason = "HTTP %d: %s" % (e.code, e.reason)
            fails.append({"name": name, "link": link, "error": reason})
            if i < 10 or len(fails) % 20 == 0:
                print("  [%d/%d] FAIL: %s - %s" % (i + 1, len(to_dl), name, reason))

        except urllib.error.URLError as e:
            reason = "URLError: " + str(e.reason)
            fails.append({"name": name, "link": link, "error": reason})
            if i < 10 or len(fails) % 20 == 0:
                print("  [%d/%d] FAIL: %s - %s" % (i + 1, len(to_dl), name, reason))

        except Exception as e:
            reason = type(e).__name__ + ": " + str(e)
            fails.append({"name": name, "link": link, "error": reason})
            if i < 10 or len(fails) % 20 == 0:
                print("  [%d/%d] FAIL: %s - %s" % (i + 1, len(to_dl), name, reason))

        time.sleep(random.uniform(0.5, 1.0))

    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print("Attempted:       %d" % len(to_dl))
    print("Successful:      %d" % ok)
    print("  Unsigned:      %d" % (ok - signed_ct))
    print("  Signed only:   %d" % signed_ct)
    print("Failed:          %d" % len(fails))
    print("Time:            %ds (%.1f min)" % (elapsed, elapsed / 60))
    if ok > 0:
        print("Avg per download:%.2fs" % (elapsed / len(to_dl)))
    print()

    final = len([f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".shortcut")])
    print("Total .shortcut files now: %d" % final)
    print()

    if fails:
        buckets = {}
        for f in fails:
            err = f["error"]
            if "HTTP 404" in err:
                b = "HTTP 404 (Not Found / Expired)"
            elif "HTTP 410" in err:
                b = "HTTP 410 (Gone)"
            elif "HTTP" in err:
                b = err.split(":")[0]
            elif "No downloadURL" in err:
                b = "No downloadURL"
            elif "URLError" in err:
                b = "Network Error"
            else:
                b = err[:50]
            buckets[b] = buckets.get(b, 0) + 1

        print("Failure breakdown:")
        for reason, count in sorted(buckets.items(), key=lambda x: -x[1]):
            print("  %4d  %s" % (count, reason))
        print()

    report = {
        "timestamp": t1.isoformat(),
        "attempted": len(to_dl),
        "successes": ok,
        "failures_count": len(fails),
        "signed_only": signed_ct,
        "elapsed_seconds": elapsed,
        "failures": fails,
    }
    with open(FAILURE_LOG, "w") as f:
        json.dump(report, f, indent=2)
    print("Failure log: " + FAILURE_LOG)
    print("Done: " + t1.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
