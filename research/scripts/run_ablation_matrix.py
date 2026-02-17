#!/usr/bin/env python3
"""
Run ablation matrix — parallel experiment execution on M1 Max.

Default max-parallel=2 on M1 Max; increase only after measured utilization.

Usage:
    uv run python research/scripts/run_ablation_matrix.py --matrix-config research/configs/ablation.yaml
    uv run python research/scripts/run_ablation_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation matrix with bounded concurrency",
    )
    parser.add_argument("--matrix-config", type=Path, required=True,
                       help="Path to ablation matrix YAML config")
    parser.add_argument("--max-parallel", type=int, default=2,
                       help="Max concurrent runs (default: 2 for M1 Max)")
    parser.add_argument("--device", type=str, default="mps",
                       choices=["mps", "cpu"],
                       help="Target device (default: mps)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from a specific run ID")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show planned runs without executing")
    args = parser.parse_args()

    raise NotImplementedError("Phase 4: ablation matrix execution")


if __name__ == "__main__":
    main()
