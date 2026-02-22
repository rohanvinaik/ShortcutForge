#!/usr/bin/env python3
"""Download exotic models for cross-architecture study.

Downloads models that aren't in the local HF cache yet.
Use --dry-run to see what would be downloaded without downloading.

Usage:
    python research/scripts/download_models.py --dry-run
    python research/scripts/download_models.py --model-id mamba-2.8b
    python research/scripts/download_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from src.model_registry import REGISTRY, get_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for cross-architecture study")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--model-id", type=str, default=None, help="Download a specific model only")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: HF default)",
    )
    args = parser.parse_args()

    if args.model_id:
        spec = get_model_spec(args.model_id)
        targets = [spec]
    else:
        targets = [s for s in REGISTRY.values() if s.needs_download and s.enabled]

    if not targets:
        print("No models need downloading.")
        return

    total_gb = sum(s.memory_gb_estimate for s in targets)
    print(f"Models to download: {len(targets)}")
    print(f"Estimated total disk usage: ~{total_gb:.0f} GB")
    print()

    for spec in targets:
        print(f"  [{spec.model_id}] {spec.display_name}")
        print(f"    Repo: {spec.hf_repo_or_path}")
        print(f"    Family: {spec.architecture_family}")
        print(f"    Params: {spec.param_count_b}B")
        print(f"    Est. disk: ~{spec.memory_gb_estimate:.0f} GB")
        if spec.trust_remote_code:
            print("    Requires: trust_remote_code=True")
        print()

    if args.dry_run:
        print("Dry run complete. No downloads executed.")
        return

    from huggingface_hub import snapshot_download

    for i, spec in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] Downloading {spec.model_id} ({spec.hf_repo_or_path})...")
        snapshot_download(
            repo_id=spec.hf_repo_or_path,
            cache_dir=args.cache_dir,
        )
        print(f"  Done: {spec.model_id}")
        print()

    print(f"All {len(targets)} models downloaded.")


if __name__ == "__main__":
    main()
