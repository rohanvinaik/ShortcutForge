#!/usr/bin/env python3
"""Phase A: Zero-shot behavioral taxonomy.

Runs each model on eval prompts, collects behavioral fingerprints,
clusters by similarity to build a cross-architecture taxonomy.

Usage:
    python research/scripts/run_phase_a.py --dry-run
    python research/scripts/run_phase_a.py --models phi-2,qwen2.5-0.5b,llama-1b
    python research/scripts/run_phase_a.py --max-prompts 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

PROJECT_ROOT = _RESEARCH_ROOT.parent

log = logging.getLogger(__name__)


def _parse_jsonl_record(line: str) -> dict | None:
    """Parse a single JSONL line, returning None for blank lines."""
    stripped = line.strip()
    if not stripped:
        return None
    return json.loads(stripped)


def _extract_prompt_from_record(record: dict) -> str | None:
    """Extract prompt text from a record.

    Supports both flat "prompt" field and chat "messages" format.
    Returns None if no prompt found.
    """
    if "prompt" in record:
        return record["prompt"]
    if "messages" not in record:
        return None
    for msg in record["messages"]:
        if msg.get("role") == "user":
            return msg["content"]
    return None


def load_eval_prompts(eval_file: Path, max_prompts: int) -> list[str]:
    """Load prompts from a JSONL file."""
    prompts: list[str] = []
    with open(eval_file) as f:
        for line in f:
            record = _parse_jsonl_record(line)
            if record is None:
                continue
            prompt = _extract_prompt_from_record(record)
            if prompt is not None:
                prompts.append(prompt)
            if len(prompts) >= max_prompts:
                break
    return prompts


def run_model(
    model_id: str,
    prompts: list[str],
    output_dir: Path,
) -> dict | None:
    """Run a single model on eval prompts and save its fingerprint.

    Returns fingerprint dict on success, None on failure.
    """
    from src.behavioral_fingerprint import BehavioralFingerprint
    from src.logit_normalizer import ActionVocab, normalize_predictions
    from src.model_adapter import load_adapter
    from src.model_registry import get_model_spec

    spec = get_model_spec(model_id)
    adapter = load_adapter(spec)

    try:
        log.info("Loading model %s", model_id)
        adapter.load()

        # Try generation with logits first, fall back to text-only
        logits = None
        try:
            texts, logits = adapter.generate_with_logits(prompts, max_tokens=256)
        except Exception as e:
            log.warning(
                "Logit extraction failed for %s: %s. Falling back to text-only.", model_id, e
            )
            texts = adapter.generate(prompts, max_tokens=256)

        # Normalize predictions
        vocab = ActionVocab()
        normalized = normalize_predictions(texts, vocab)

        # Build fingerprint
        if logits is not None:
            fp = BehavioralFingerprint.from_outputs(
                experiment_id=model_id,
                step=0,
                output_logits=logits,
                action_predictions=normalized,
                probe_labels=[f"prompt_{i}" for i in range(len(normalized))],
            )
        else:
            fp = BehavioralFingerprint.from_text_only(
                experiment_id=model_id,
                step=0,
                action_predictions=normalized,
                probe_labels=[f"prompt_{i}" for i in range(len(normalized))],
            )

        # Save fingerprint
        model_dir = output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        fp.save(model_dir / "fingerprint.json")
        log.info("Saved fingerprint for %s", model_id)

        return {"model_id": model_id, "status": "success", "n_predictions": len(normalized)}

    except Exception as e:
        log.error("Failed to process model %s: %s", model_id, e)
        return {"model_id": model_id, "status": "error", "error": str(e)}

    finally:
        adapter.unload()


def run_taxonomy(model_ids: list[str], output_dir: Path) -> None:
    """Load saved fingerprints and build taxonomy report."""
    from src.behavioral_fingerprint import BehavioralFingerprint
    from src.fingerprint_taxonomy import build_taxonomy
    from src.model_registry import get_model_spec

    fingerprints = []
    specs = []
    for mid in model_ids:
        fp_path = output_dir / mid / "fingerprint.json"
        if not fp_path.exists():
            log.warning("No fingerprint found for %s, skipping", mid)
            continue
        fingerprints.append(BehavioralFingerprint.load(fp_path))
        specs.append(get_model_spec(mid))

    if len(fingerprints) < 2:
        log.warning("Need at least 2 fingerprints for taxonomy, got %d", len(fingerprints))
        return

    report = build_taxonomy(fingerprints, specs)

    # Save report
    report_path = output_dir / "taxonomy_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Phase B expects model_id -> cluster_id mapping.
    model_to_cluster: dict[str, int] = {}
    for cluster_id, members in report.clusters.items():
        for mid in members:
            model_to_cluster[mid] = int(cluster_id)
    clusters_path = output_dir / "taxonomy_clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(model_to_cluster, f, indent=2)

    md_path = output_dir / "taxonomy_report.md"
    md_path.write_text(report.to_markdown())

    print("\n" + report.to_markdown())
    print(f"\nReport saved to: {report_path}")
    print(f"Cluster map saved to: {clusters_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase A."""
    parser = argparse.ArgumentParser(description="Phase A: Zero-shot behavioral taxonomy")
    parser.add_argument("--dry-run", action="store_true", help="List models and exit")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RESEARCH_ROOT / "results" / "phase_a",
        help="Output directory for results",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_eval.jsonl",
        help="Path to eval JSONL file",
    )
    parser.add_argument("--max-prompts", type=int, default=100, help="Max number of eval prompts")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def _print_dry_run(model_ids: list[str], args: argparse.Namespace) -> None:
    """Display dry-run summary of models and configuration."""
    from src.model_registry import get_model_spec

    print(f"Phase A: {len(model_ids)} models")
    print()
    for mid in model_ids:
        spec = get_model_spec(mid)
        dl = " (needs download)" if spec.needs_download else ""
        print(
            f"  [{spec.model_id}] {spec.display_name} "
            f"({spec.architecture_family}, {spec.param_count_b}B){dl}"
        )
    print()
    print(f"Eval file: {args.eval_file}")
    print(f"Max prompts: {args.max_prompts}")
    print(f"Output dir: {args.output_dir}")
    print("\nDry run complete.")


def _save_summary_and_taxonomy(
    model_ids: list[str],
    results: list[dict],
    prompts: list[str],
    output_dir: Path,
    elapsed: float,
) -> None:
    """Save run summary JSON and build taxonomy from successful runs."""
    summary = {
        "model_ids": model_ids,
        "n_prompts": len(prompts),
        "results": results,
        "elapsed_seconds": elapsed,
    }
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    successful = [r["model_id"] for r in results if r["status"] == "success"]
    if len(successful) >= 2:
        print("\n=== Building Taxonomy ===")
        run_taxonomy(successful, output_dir)
    else:
        print(f"\nOnly {len(successful)} successful runs, need >= 2 for taxonomy.")


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine model list
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        from src.model_registry import get_available

        model_ids = [s.model_id for s in get_available()]

    if args.dry_run:
        _print_dry_run(model_ids, args)
        return

    # Load prompts
    prompts = load_eval_prompts(args.eval_file, args.max_prompts)
    print(f"Loaded {len(prompts)} eval prompts from {args.eval_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run each model
    results = []
    start = time.time()

    for i, mid in enumerate(model_ids, 1):
        print(f"\n[{i}/{len(model_ids)}] Processing {mid}...")
        result = run_model(mid, prompts, args.output_dir)
        if result:
            results.append(result)
            print(f"  Status: {result['status']}")

    elapsed = time.time() - start
    _save_summary_and_taxonomy(model_ids, results, prompts, args.output_dir, elapsed)
    print(f"\nPhase A complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
