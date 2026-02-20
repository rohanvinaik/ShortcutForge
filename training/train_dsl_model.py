#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 8B Instruct on ShortcutDSL via Tinker.

Uses LoRA fine-tuning with remote GPU through Tinker API.
Trains only on the assistant (DSL) output using LAST_ASSISTANT_MESSAGE.

Usage:
    # Baseline training (~1,671 examples)
    python scripts/train_dsl_model.py --file-path training_data/shortcutdsl_train.jsonl

    # Expanded training (~5,000 examples)
    python scripts/train_dsl_model.py --file-path training_data/shortcutdsl_train_expanded.jsonl --num-epochs 2

    # Mini run for adapter conversion testing
    python scripts/train_dsl_model.py --file-path training_data/shortcutdsl_train.jsonl --mini

Prerequisites:
    TINKER_API_KEY environment variable must be set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LOG_PATH = "/tmp/shortcutforge-training"
DEFAULT_FILE_PATH = "training_data/shortcutdsl_train.jsonl"


def build_config(
    file_path: str,
    log_path: str = DEFAULT_LOG_PATH,
    model_name: str = DEFAULT_MODEL,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    batch_size: int = 8,
    max_length: int = 4096,
    save_every: int = 500,
    eval_every: int = 100,
    mini: bool = False,
) -> train.Config:
    """Build a Tinker training config for ShortcutDSL fine-tuning."""
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    if mini:
        # Mini run: 1 epoch, frequent saves, for adapter conversion testing
        num_epochs = 1
        save_every = 50
        eval_every = 25

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=file_path,
        test_size=0,  # We use our own eval split (leakage-safe)
    )

    blueprint = chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": model_name,
            "dataset_builder": dataset_builder,
            "learning_rate": learning_rate,
            "lr_schedule": "cosine",
            "num_epochs": num_epochs,
            "lora_rank": lora_rank,
            "save_every": save_every,
            "eval_every": eval_every,
        }
    )

    return blueprint.make()


def main():
    parser = argparse.ArgumentParser(
        prog="train_dsl_model",
        description="Fine-tune Llama 3.1 8B on ShortcutDSL via Tinker",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=DEFAULT_FILE_PATH,
        help=f"Training JSONL file (default: {DEFAULT_FILE_PATH})",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=DEFAULT_LOG_PATH,
        help=f"Tinker log/checkpoint directory (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max sequence length (default: 4096)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluate every N steps (default: 100)",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Mini run for testing (1 epoch, frequent saves)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build config and print it, don't train",
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("TINKER_API_KEY"):
        print("Error: TINKER_API_KEY not set", file=sys.stderr)
        print("Get your key from your Tinker account and export it.", file=sys.stderr)
        sys.exit(1)

    # Resolve file path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    file_path = (
        args.file_path
        if os.path.isabs(args.file_path)
        else str(project_root / args.file_path)
    )

    if not os.path.exists(file_path):
        print(f"Error: Training file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Count examples
    with open(file_path) as f:
        n_examples = sum(1 for _ in f)

    print("\nShortcutForge: Fine-tuning DSL model\n")
    print(f"  Model:      {args.model_name}")
    print(f"  Data:       {file_path} ({n_examples} examples)")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LoRA rank:  {args.lora_rank}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print(f"  Log path:   {args.log_path}")
    if args.mini:
        print("  Mode:       MINI (1 epoch, frequent saves)")
    print()

    config = build_config(
        file_path=file_path,
        log_path=args.log_path,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_every=args.save_every,
        eval_every=args.eval_every,
        mini=args.mini,
    )

    if args.check_only:
        print("  Config built successfully. Exiting (--check-only).")
        return

    # Check log directory
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    print("  Starting training...\n")
    asyncio.run(train.main(config))
    print("\n  Training complete.")
    print(f"  Checkpoints saved to: {args.log_path}")


if __name__ == "__main__":
    main()
