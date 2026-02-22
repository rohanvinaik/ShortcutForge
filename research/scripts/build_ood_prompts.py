#!/usr/bin/env python3
"""
Build out-of-domain prompt set for domain gate training.

Combines in-domain samples from training data with synthetically
generated OOD prompts from various categories.

Usage:
    uv run python research/scripts/build_ood_prompts.py -v
    uv run python research/scripts/build_ood_prompts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from research.src.contracts import OODPrompt  # noqa: E402

# ---------------------------------------------------------------------------
# OOD Templates (deterministic, no LLM calls)
# ---------------------------------------------------------------------------

OOD_TEMPLATES: dict[str, list[str]] = {
    "general_chat": [
        "What's the weather like today?",
        "Tell me a joke about programming",
        "How are you feeling?",
        "What's your favorite movie?",
        "Can you recommend a good restaurant?",
        "What time is it in Tokyo?",
        "Who won the World Series last year?",
        "What's the meaning of life?",
        "Tell me something interesting about space",
        "What should I have for dinner tonight?",
        "How do I tie a tie?",
        "What's the capital of France?",
        "Explain quantum physics to me",
        "What's your opinion on artificial intelligence?",
        "Tell me a fun fact about dolphins",
        "What's the best way to learn a new language?",
        "How do I make friends as an adult?",
        "What are some good books to read?",
        "Describe the color blue to a blind person",
        "What happened in the news today?",
        "How do I get better at chess?",
        "What are the benefits of meditation?",
    ],
    "code_generation": [
        "Write a Python function that sorts a list",
        "Create a JavaScript class for a shopping cart",
        "Implement a binary search tree in Java",
        "Write a SQL query to find duplicate records",
        "Create a React component for a login form",
        "Implement a linked list in C++",
        "Write a Python decorator for caching",
        "Create a REST API endpoint in Flask",
        "Write a regex to validate email addresses",
        "Implement merge sort in Python",
        "Create a TypeScript interface for a user profile",
        "Write a bash script to find large files",
        "Implement a stack using two queues",
        "Create a CSS grid layout for a dashboard",
        "Write a Go function for concurrent HTTP requests",
        "Implement a rate limiter in Python",
        "Create a database migration script",
        "Write a unit test for a calculator class",
        "Implement a pub/sub system in Node.js",
        "Create a Docker compose file for a web app",
        "Write a GraphQL schema for a blog",
        "Implement an LRU cache in Python",
    ],
    "creative_writing": [
        "Write a poem about the ocean",
        "Compose a haiku about autumn",
        "Write a short story about a time traveler",
        "Create a limerick about a lazy cat",
        "Write a love letter from one planet to another",
        "Compose a sonnet about technology",
        "Write a fairy tale set in modern times",
        "Create a monologue for a villain",
        "Write a song about Monday mornings",
        "Compose a eulogy for a fictional character",
        "Write a children's story about a brave mouse",
        "Create a mystery story opening paragraph",
        "Write a science fiction story about first contact",
        "Compose a ballad about a lost treasure",
        "Write a horror story in exactly 100 words",
        "Create dialogue between a robot and a human",
        "Write a travel blog post about an imaginary place",
        "Compose a graduation speech for aliens",
        "Write a recipe as if it were a love story",
        "Create a news article from the year 3000",
        "Write a diary entry from a cat's perspective",
    ],
    "math_homework": [
        "Solve the equation 2x + 5 = 15",
        "What is the derivative of x^3?",
        "Calculate the area of a circle with radius 5",
        "Find the integral of sin(x)dx",
        "What is the probability of rolling two sixes?",
        "Simplify the expression (3x^2 + 2x) / x",
        "Find the eigenvalues of a 2x2 matrix",
        "Calculate the standard deviation of 1,2,3,4,5",
        "Prove that the square root of 2 is irrational",
        "What is 15% of 240?",
        "Convert 72 degrees Fahrenheit to Celsius",
        "Find the GCD of 48 and 18",
        "Calculate the volume of a sphere with radius 3",
        "What is the sum of the first 100 natural numbers?",
        "Solve the system of equations: x+y=10, x-y=4",
        "Find the limit of (sin x)/x as x approaches 0",
        "Calculate compound interest on $1000 at 5% for 3 years",
        "What is the factorial of 10?",
        "Find the slope of the line passing through (2,3) and (5,9)",
        "Convert the binary number 101101 to decimal",
        "What is the determinant of a 3x3 identity matrix?",
    ],
    "other_automation": [
        "Create a Zapier workflow to sync contacts",
        "Build an IFTTT applet for smart home",
        "Write a bash script to backup files",
        "Set up a cron job to run every hour",
        "Create a Makefile for a C project",
        "Write an Ansible playbook for server setup",
        "Create a GitHub Actions workflow for CI/CD",
        "Build a Terraform config for AWS",
        "Write a Kubernetes deployment manifest",
        "Create a Jenkins pipeline for testing",
        "Set up a Prometheus alerting rule",
        "Write a Puppet manifest for package management",
        "Create a Chef recipe for web server setup",
        "Build a CircleCI config for a Node project",
        "Write a Gradle build script",
        "Create a systemd service file",
        "Write a PowerShell script to manage users",
        "Build a Slack bot integration",
        "Create a Datadog monitor for CPU usage",
        "Write a CloudFormation template",
        "Set up a GitLab CI pipeline",
    ],
}

# Light variation prefixes and suffixes
PREFIXES = ["", "Please ", "Can you ", "I need you to ", "Help me "]
SUFFIXES = ["", " please", " for me", " right now"]


# ---------------------------------------------------------------------------
# Sub-functions
# ---------------------------------------------------------------------------


def _generate_ood_prompts(ood_n: int, verbose: bool) -> list[OODPrompt]:
    """Generate OOD prompts from templates with prefix/suffix variation, then sample."""
    print("=== Generating OOD prompts ===")
    all_ood_prompts: list[OODPrompt] = []

    for category, templates in OOD_TEMPLATES.items():
        category_prompts = _expand_category(category, templates)
        all_ood_prompts.extend(category_prompts)
        if verbose:
            print(f"  {category}: {len(category_prompts)} variants")

    print(f"  Total OOD pool: {len(all_ood_prompts)} prompts")

    if len(all_ood_prompts) > ood_n:
        ood_sample = random.sample(all_ood_prompts, ood_n)
    else:
        ood_sample = all_ood_prompts
    print(f"  Sampled {len(ood_sample)} OOD prompts")
    return ood_sample


def _expand_category(category: str, templates: list[str]) -> list[OODPrompt]:
    """Expand a single OOD category's templates with prefix/suffix variations."""
    prompts: list[OODPrompt] = []
    for template in templates:
        for prefix in PREFIXES:
            for suffix in SUFFIXES:
                prompt_text = f"{prefix}{template}{suffix}".strip()
                # Lowercase the first character after prefix if prefix is non-empty
                if prefix and prompt_text:
                    prompt_text = prefix + template[0].lower() + template[1:] + suffix
                    prompt_text = prompt_text.strip()

                prompts.append(
                    OODPrompt(
                        prompt=prompt_text,
                        label="ood",
                        category=category,
                        source="synthetic",
                    )
                )
    return prompts


def _load_in_domain_prompts(train_in: Path, in_domain_n: int) -> list[OODPrompt]:
    """Load in-domain prompts from training data and sample."""
    print("\n=== Loading in-domain prompts ===")

    if not train_in.exists():
        print(f"  WARNING: {train_in} not found, skipping in-domain prompts")
        return []

    raw_prompts = _extract_user_prompts(train_in)
    print(f"  Loaded {len(raw_prompts)} in-domain prompts from training data")

    if len(raw_prompts) > in_domain_n:
        sampled = random.sample(raw_prompts, in_domain_n)
    else:
        sampled = raw_prompts

    in_domain_prompts = [
        OODPrompt(
            prompt=prompt_text,
            label="in_domain",
            category="apple_shortcuts",
            source="seed_file",
        )
        for prompt_text in sampled
    ]
    print(f"  Sampled {len(in_domain_prompts)} in-domain prompts")
    return in_domain_prompts


def _parse_jsonl_record(line: str) -> dict | None:
    """Safe JSON decode with error handling.

    Returns parsed dict, or None on failure (invalid JSON, empty/whitespace lines).
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_user_message(record: dict) -> str | None:
    """Extract the first user-role content from a messages record.

    Returns the stripped user message text, or None if no user message found.
    """
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            prompt_text = msg.get("content", "").strip()
            return prompt_text if prompt_text else None
    return None


def _extract_user_prompts(train_in: Path) -> list[str]:
    """Extract user-role prompt strings from a training JSONL file."""
    raw_prompts: list[str] = []
    with open(train_in) as f:
        for line in f:
            record = _parse_jsonl_record(line)
            if record is None:
                continue
            prompt = _extract_user_message(record)
            if prompt:
                raw_prompts.append(prompt)
    return raw_prompts


def _load_seed_file(seed_file: Path | None) -> list[OODPrompt]:
    """Load optional seed file with manual OOD examples."""
    if not seed_file or not seed_file.exists():
        return []

    print(f"\n=== Loading seed file: {seed_file} ===")
    seed_entries: list[OODPrompt] = []
    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                seed_entries.append(OODPrompt.from_dict(d))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Loaded {len(seed_entries)} seed entries")
    return seed_entries


def _combine_and_save(combined: list[OODPrompt], out: Path, dry_run: bool) -> None:
    """Shuffle, print summary, and optionally write the combined prompt set."""
    random.shuffle(combined)

    # Category and label breakdown
    category_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {"in_domain": 0, "ood": 0}
    for entry in combined:
        category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        label_counts[entry.label] = label_counts.get(entry.label, 0) + 1

    print("\n=== Summary ===")
    print(f"  Total entries: {len(combined)}")
    print("  Labels:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    print("  Categories:")
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat}: {count}")

    if dry_run:
        print("\n  --dry-run: skipping write")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in combined:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Wrote {len(combined)} entries to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build OOD prompt set for domain gate training",
    )
    parser.add_argument(
        "--train-in",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_train_expanded.jsonl",
        help="Training JSONL (in-domain source)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "references" / "ood_prompt_set.jsonl",
        help="Output OOD prompt set JSONL",
    )
    parser.add_argument(
        "--in-domain-n",
        type=int,
        default=500,
        help="Number of in-domain samples to include",
    )
    parser.add_argument("--ood-n", type=int, default=500, help="Number of OOD samples to generate")
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Optional seed file with manual OOD examples",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    random.seed(args.seed)

    ood_prompts = _generate_ood_prompts(args.ood_n, args.verbose)
    in_domain_prompts = _load_in_domain_prompts(args.train_in, args.in_domain_n)
    seed_entries = _load_seed_file(args.seed_file)

    combined: list[OODPrompt] = ood_prompts + in_domain_prompts + seed_entries
    _combine_and_save(combined, args.out, args.dry_run)


if __name__ == "__main__":
    main()
