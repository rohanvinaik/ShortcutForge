#!/usr/bin/env python3
"""
Generate a diverse set of test prompts for ShortcutForge evaluation.

Creates prompts across complexity tiers and action categories to stress-test
the generation pipeline beyond the frozen eval set.

Usage:
    python scripts/generate_test_prompts.py --count 50 --output training_data/new_eval_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ── Prompt Templates ──────────────────────────────────────────────────

# Simple (single action, short)
SIMPLE_PROMPTS = [
    "Open the Settings app.",
    "Set the screen brightness to 75 percent.",
    "Toggle Do Not Disturb.",
    "Turn on the flashlight.",
    "Play the current song in Apple Music.",
    "Show the current battery level.",
    "Get the current weather.",
    "Open Safari and go to google.com.",
    "Set a timer for 5 minutes.",
    "Take a screenshot.",
    "Launch the Camera app.",
    "Turn on Bluetooth.",
    "Set the volume to 50 percent.",
    "Enable Low Power Mode.",
    "Open the Health app.",
    "Get the current date and time.",
    "Start a stopwatch.",
    "Turn off Wi-Fi.",
    "Open the Calendar app.",
    "Copy the current clipboard contents to a note.",
]

# Medium (multi-step, some logic)
MEDIUM_PROMPTS = [
    "Ask for a name, then create a new contact with that name.",
    "Get the current weather and send it as a text message to my wife.",
    "Take a photo with the rear camera and save it to the Photos library.",
    "Ask the user for a URL, then open it in Safari.",
    "Get the current date, format it as YYYY-MM-DD, and copy it to the clipboard.",
    "Search for nearby coffee shops and show the results on a map.",
    "Get the latest photo from my library and share it via AirDrop.",
    "Create a new reminder called 'Buy groceries' due tomorrow at 9 AM.",
    "Ask for text input, then speak it aloud using Siri's voice.",
    "Get my current location and save the address to a note called 'Saved Locations'.",
    "Scan a QR code and open the resulting URL in Safari.",
    "Record a voice memo for 30 seconds and save it.",
    "Get the current song playing and add it to a playlist called 'Favorites'.",
    "Ask for a number and calculate its square root, then show the result.",
    "Download the contents of a URL and save it to Files.",
    "Take a selfie and set it as the wallpaper.",
    "Get the word count of text on the clipboard and show it in an alert.",
    "Search for a specific app in the App Store.",
    "Create a calendar event for tomorrow at 2 PM called 'Team Meeting'.",
    "Ask for a search query and open Google with that query in Safari.",
]

# Complex (menus, conditionals, loops, multi-step)
COMPLEX_PROMPTS = [
    "Present a menu with options 'Work', 'Home', and 'Gym'. Based on the selection, set the appropriate Focus mode.",
    "Ask the user to choose from a list of playlists, then shuffle and play the selected one.",
    "Get all reminders due today, format them as a bulleted list, and send the list via iMessage to a chosen contact.",
    "Scan a document using the camera, convert it to PDF, and save it to iCloud Drive with today's date in the filename.",
    "Create a shortcut that presents a menu to choose between Celsius and Fahrenheit, asks for a temperature, converts it, and shows the result.",
    "Check if the current time is before noon. If so, say 'Good morning' and show today's calendar events. Otherwise, say 'Good afternoon' and show tomorrow's events.",
    "Ask for a list of items separated by commas, then iterate through each item and create a reminder for it.",
    "Get the current battery level. If it's below 20%, enable Low Power Mode and reduce brightness to 30%. Otherwise, show a notification that battery is fine.",
    "Present a menu of social media apps (Twitter, Instagram, Reddit). Open the selected app, or if 'Post Text' is chosen, ask for text and share it.",
    "Get the weather forecast for the week, check each day, and if rain is predicted for any day, add a reminder to bring an umbrella.",
    "Create a workout logger that asks for exercise type from a menu, then asks for sets, reps, and weight, and saves it all to a Health note.",
    "Build a flashcard quiz that shows a random question from a dictionary, waits for the user's answer, checks if it matches, and keeps score.",
    "Get all photos from today, let the user select which ones to keep, and create a collage or album from the selection.",
    "Create a daily journal entry that asks for mood (using a menu with emoji options), a text entry for notes, and appends everything to a running note with the date.",
    "Build a unit converter with a menu for category (length, weight, temperature), then sub-menus for specific units, and perform the conversion.",
    "Check the calendar for meetings in the next hour. If there's a meeting, set Do Not Disturb and send an auto-reply text to recent callers.",
    "Get a list of files in a specific iCloud folder, present them as a menu, and let the user preview, share, or delete the selected file.",
    "Create a morning routine shortcut that turns off the alarm, shows the weather, reads the top 3 news headlines, and starts a playlist.",
    "Build a shopping list manager: present options to add item, view list, clear list, or share list. Each option performs the corresponding action.",
    "Create a Pomodoro timer that runs for 25 minutes, then asks if you want a 5-minute break or 15-minute break, and repeats the cycle.",
]

# Edge cases / tricky prompts
EDGE_CASE_PROMPTS = [
    "Create an empty shortcut that does nothing.",
    "Open an app called 'My App With Spaces' in the name.",
    "Set a variable to the result of a math expression: (10 + 5) * 3.",
    "Create a shortcut with a very long name: 'This Is My Super Long Shortcut Name That Tests Edge Cases'.",
    "Ask for input that contains special characters like quotes and backslashes.",
    "Create a nested if-else: if battery > 80, check if Wi-Fi is on; if both true, start backup.",
    "Use a repeat loop that runs exactly 1 time.",
    "Create a menu with only one option.",
    "Get text from the clipboard and replace all occurrences of 'old' with 'new'.",
    "Create a shortcut that uses both a dictionary and a list together.",
]

# Third-party app prompts (tests unknown action handling)
THIRD_PARTY_PROMPTS = [
    "Open the Spotify app and play my Discover Weekly playlist.",
    "Use the Overcast app to play the most recent podcast episode.",
    "Open Notion and create a new page in my workspace.",
    "Use Bear to create a new note with today's date as the title.",
    "Open the Shortcuts app for Scriptable and run a specific script.",
    "Use Drafts to create a new draft with the clipboard contents.",
    "Open Fantastical and create a new event.",
    "Use Carrot Weather to get a snarky weather forecast.",
    "Open 1Password and search for a specific login.",
    "Use Toolbox Pro to get the device model name.",
]


def generate_eval_set(
    count: int = 50,
    seed: int = 42,
    system_prompt: str | None = None,
) -> list[dict]:
    """Generate a diverse eval set from prompt templates.

    Each entry matches the training JSONL format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ],
        "shortcut_id": "generated_N",
    }
    """
    rng = random.Random(seed)

    # Build pool with category tags
    pool: list[tuple[str, str]] = []
    for p in SIMPLE_PROMPTS:
        pool.append((p, "simple"))
    for p in MEDIUM_PROMPTS:
        pool.append((p, "medium"))
    for p in COMPLEX_PROMPTS:
        pool.append((p, "complex"))
    for p in EDGE_CASE_PROMPTS:
        pool.append((p, "edge_case"))
    for p in THIRD_PARTY_PROMPTS:
        pool.append((p, "third_party"))

    # Get system prompt
    if system_prompt is None:
        from generate_prompt import build_system_prompt
        system_prompt = build_system_prompt()

    # Sample with distribution: ~30% simple, ~30% medium, ~25% complex, ~10% edge, ~5% third-party
    # But cap at available prompts per category
    categories = {
        "simple": [p for p, c in pool if c == "simple"],
        "medium": [p for p, c in pool if c == "medium"],
        "complex": [p for p, c in pool if c == "complex"],
        "edge_case": [p for p, c in pool if c == "edge_case"],
        "third_party": [p for p, c in pool if c == "third_party"],
    }

    target_distribution = {
        "simple": 0.30,
        "medium": 0.30,
        "complex": 0.25,
        "edge_case": 0.10,
        "third_party": 0.05,
    }

    selected: list[tuple[str, str]] = []
    for cat, fraction in target_distribution.items():
        n = min(int(count * fraction), len(categories[cat]))
        selected.extend((p, cat) for p in rng.sample(categories[cat], n))

    # Fill remaining from all categories
    remaining = count - len(selected)
    if remaining > 0:
        used_prompts = {p for p, _ in selected}
        available = [(p, c) for p, c in pool if p not in used_prompts]
        rng.shuffle(available)
        selected.extend(available[:remaining])

    rng.shuffle(selected)

    examples = []
    for i, (prompt, category) in enumerate(selected):
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "shortcut_id": f"generated_{i}",
            "category": category,
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse test prompts for ShortcutForge evaluation",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of prompts to generate (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="training_data/new_eval_prompts.jsonl",
        help="Output JSONL file path",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)

    print(f"Generating {args.count} test prompts (seed={args.seed})...")
    examples = generate_eval_set(count=args.count, seed=args.seed)

    # Write
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    cats = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"Written {len(examples)} prompts to {output_path}")
    print(f"Distribution:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n} ({n/len(examples)*100:.0f}%)")


if __name__ == "__main__":
    main()
