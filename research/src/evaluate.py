"""
Evaluation harness for Balanced Sashimi checkpoints.

Usage:
    uv run python research/src/evaluate.py \
      --config research/configs/base.yaml \
      --checkpoint path/to/ckpt.pt \
      --eval-file training_data/typed_ir_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.behavioral_fingerprint import BehavioralFingerprint
except ImportError:  # pragma: no cover - fallback for direct module execution
    from research.src.behavioral_fingerprint import BehavioralFingerprint


class _VocabInfo(NamedTuple):
    """Vocabulary data loaded from config paths."""

    tier1_vocab: dict[str, int]
    tier1_idx2token: dict[int, str]
    tier2_vocab_size: int


class _EvalPipeline(NamedTuple):
    """Bundle of model components used during evaluation."""

    encoder: Any
    domain_gate: Any
    intent_extractor: Any
    bridge: Any
    decoder: Any
    checkpoint_meta: dict[str, Any]


def _load_vocabs(config: dict) -> _VocabInfo:
    """Load tier1 and tier2 vocabularies from config paths."""
    data_cfg = config["data"]
    tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
    tier2_vocab_dir = PROJECT_ROOT / data_cfg["tier2_vocab_dir"]
    tier2_fallback = json.loads((tier2_vocab_dir / "_global_fallback.json").read_text())
    tier1_idx2token = {v: k for k, v in tier1_vocab.items()}
    return _VocabInfo(
        tier1_vocab=tier1_vocab,
        tier1_idx2token=tier1_idx2token,
        tier2_vocab_size=len(tier2_fallback),
    )


def _build_pipeline(config: dict, checkpoint_path: Path, vocab: _VocabInfo) -> _EvalPipeline:
    """Reconstruct the model pipeline from config and checkpoint weights."""
    import torch

    try:
        from src.bridge import InformationBridge
        from src.domain_gate import DomainGate
        from src.encoder import PromptEncoder
        from src.intent_extractor import IntentExtractor
        from src.ternary_decoder import TernaryDecoder
    except ImportError:  # pragma: no cover - fallback for direct module execution
        from research.src.bridge import InformationBridge
        from research.src.domain_gate import DomainGate
        from research.src.encoder import PromptEncoder
        from research.src.intent_extractor import IntentExtractor
        from research.src.ternary_decoder import TernaryDecoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]
    decoder_cfg = model_cfg["decoder"]

    encoder = PromptEncoder(
        model_name=model_cfg["encoder"]["model_name"],
        device="cpu",
    )
    domain_gate = DomainGate(
        input_dim=model_cfg["encoder"]["output_dim"],
        hidden_dim=model_cfg["domain_gate"]["hidden_dim"],
    )
    intent_extractor = IntentExtractor(
        input_dim=model_cfg["encoder"]["output_dim"],
        frame_dim=model_cfg["intent_extractor"]["frame_dim"],
    )
    bridge = InformationBridge(
        input_dim=model_cfg["intent_extractor"]["frame_dim"],
        bridge_dim=model_cfg["bridge"]["bridge_dim"],
    )
    decoder = TernaryDecoder(
        input_dim=model_cfg["bridge"]["bridge_dim"],
        hidden_dim=decoder_cfg["hidden_dim"],
        tier1_vocab_size=len(vocab.tier1_vocab),
        tier2_vocab_size=vocab.tier2_vocab_size,
        num_layers=decoder_cfg["num_layers"],
        ternary_enabled=decoder_cfg.get("ternary_enabled", True),
        partial_ternary=decoder_cfg.get("partial_ternary", False),
    )

    domain_gate.load_state_dict(ckpt["domain_gate"])
    intent_extractor.load_state_dict(ckpt["intent_extractor"])
    bridge.load_state_dict(ckpt["bridge"])
    decoder.load_state_dict(ckpt["decoder"])

    for module in [domain_gate, intent_extractor, bridge, decoder]:
        module.eval()

    return _EvalPipeline(
        encoder=encoder,
        domain_gate=domain_gate,
        intent_extractor=intent_extractor,
        bridge=bridge,
        decoder=decoder,
        checkpoint_meta=ckpt,
    )


def _target_tier1_token(example: Any) -> str:
    tokens = getattr(example, "tier1_tokens", [])
    if not tokens:
        return "<UNK>"
    return tokens[1] if len(tokens) > 1 else tokens[0]


def _infer_domain(example: Any) -> str:
    prompt = (getattr(example, "prompt", "") or "").lower()
    if any(k in prompt for k in ("health", "workout", "steps", "sleep", "caffeine")):
        return "health_logger"
    if any(k in prompt for k in ("api", "http", "json", "request", "endpoint")):
        return "api_pagination_fetcher"
    if any(k in prompt for k in ("calendar", "meeting", "event")):
        return "calendar_triage"
    if any(k in prompt for k in ("clipboard", "copy", "paste")):
        return "clipboard_utility"
    if any(k in prompt for k in ("file", "folder", "document", "pdf")):
        return "file_router"
    if any(k in prompt for k in ("music", "photo", "video", "media")):
        return "media_metadata_pipeline"
    if any(k in prompt for k in ("morning", "alarm", "wake", "routine")):
        return "morning_routine"
    if any(k in prompt for k in ("share", "sheet", "clean", "sanitize")):
        return "share_sheet_text_cleaner"
    return "general"


def _render_predicted_dsl(predicted_action: str, shortcut_name: str = "Generated") -> str:
    """Render minimal deterministic DSL for parser/validator/compiler checks."""
    name = (shortcut_name or "Generated").replace('"', '\\"')
    return f'SHORTCUT "{name}"\nACTION {predicted_action}\nENDSHORTCUT\n'


def _evaluate_dsl(dsl_text: str) -> dict[str, bool]:
    """Run parse/validate/compile strict+permissive checks on generated DSL."""
    try:
        from dsl_bridge import compile_ir
        from dsl_parser import parse_dsl
        from dsl_validator import validate_ir
    except ImportError as e:
        return {
            "parsed": False,
            "validated_strict": False,
            "validated_permissive": False,
            "compiled_strict": False,
            "compiled_permissive": False,
            "runtime_unverified": False,
            "import_error": True,
            "error": str(e),
        }

    parsed = validated_strict = validated_permissive = False
    compiled_strict = compiled_permissive = False
    runtime_unverified = False
    has_compiler_risk = False
    error = ""

    try:
        ir = parse_dsl(dsl_text)
        parsed = True
    except Exception as e:  # pragma: no cover - parse failures are expected examples
        return {
            "parsed": False,
            "validated_strict": False,
            "validated_permissive": False,
            "compiled_strict": False,
            "compiled_permissive": False,
            "runtime_unverified": False,
            "import_error": False,
            "error": f"parse: {e}",
        }

    try:
        strict_validation = validate_ir(ir, strict=True)
        validated_strict = not strict_validation.errors
    except Exception as e:
        error = f"validate_strict: {e}"

    try:
        permissive_validation = validate_ir(ir, strict=False)
        validated_permissive = not permissive_validation.errors
        has_compiler_risk = any(w.category == "compiler_risk" for w in permissive_validation.warnings)
    except Exception as e:
        if not error:
            error = f"validate_permissive: {e}"

    if validated_strict:
        try:
            compile_ir(ir)
            compiled_strict = True
        except Exception as e:
            if not error:
                error = f"compile_strict: {e}"

    if validated_permissive:
        if compiled_strict:
            compiled_permissive = True
        else:
            try:
                compile_ir(ir)
                compiled_permissive = True
            except Exception as e:
                if not error:
                    error = f"compile_permissive: {e}"

    runtime_unverified = compiled_permissive and (not compiled_strict or has_compiler_risk)

    return {
        "parsed": parsed,
        "validated_strict": validated_strict,
        "validated_permissive": validated_permissive,
        "compiled_strict": compiled_strict,
        "compiled_permissive": compiled_permissive,
        "runtime_unverified": runtime_unverified,
        "import_error": False,
        "error": error,
    }


def _attach_fingerprint(
    results: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    logits_rows: list[np.ndarray],
    action_predictions: list[str],
    probe_labels: list[str],
    eval_cfg: dict[str, Any],
    output_json: Path | None,
) -> None:
    """Compute behavioral fingerprint and attach metrics to results dict."""
    if not logits_rows:
        return

    fingerprint = BehavioralFingerprint.from_outputs(
        experiment_id=checkpoint_meta.get("run_id", "unknown"),
        step=checkpoint_meta.get("step", 0),
        output_logits=np.stack(logits_rows),
        action_predictions=action_predictions,
        probe_labels=probe_labels,
    )
    results["fingerprint_entropy"] = fingerprint.action_entropy
    results["fingerprint_discreteness"] = fingerprint.discreteness_score

    if eval_cfg.get("fingerprint_save", False) and output_json:
        fp_path = Path(output_json).with_suffix(".fingerprint.json")
        fingerprint.save(fp_path)
        print(f"Fingerprint written to {fp_path}")


def _write_results(results: dict, output_json: Path) -> None:
    """Serialize evaluation results to JSON file."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {output_path}")


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    eval_file: Path,
    output_json: Path | None = None,
) -> dict:
    """Evaluate a trained checkpoint against typed IR eval data."""
    import torch

    try:
        from src.data import load_typed_ir_jsonl
    except ImportError:  # pragma: no cover - fallback for direct module execution
        from research.src.data import load_typed_ir_jsonl

    examples = load_typed_ir_jsonl(eval_file)
    vocab = _load_vocabs(config)
    pipeline = _build_pipeline(config, checkpoint_path, vocab)
    gate_threshold = float(config.get("evaluation", {}).get("gate_threshold", 0.5))
    eval_cfg = config.get("evaluation", {})
    fingerprint_enabled = bool(eval_cfg.get("fingerprint_enabled", False))

    total = len(examples)
    gate_correct = 0
    tier1_exact = 0
    parse_pass = 0
    validate_strict_pass = 0
    validate_perm_pass = 0
    compile_strict_pass = 0
    compile_perm_pass = 0
    runtime_unverified = 0

    domain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "compiled_strict": 0})
    action_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    output_errors: list[dict[str, str]] = []

    logits_rows: list[np.ndarray] = []
    action_predictions: list[str] = []
    probe_labels: list[str] = []
    tier1_targets = []

    print(f"Evaluating {total} examples...")
    with torch.no_grad():
        for i, ex in enumerate(examples):
            prompt = ex.prompt
            expected_action = _target_tier1_token(ex)
            domain = _infer_domain(ex)
            tier1_targets.append(expected_action)

            emb = pipeline.encoder.encode([prompt])
            gate_prob = torch.sigmoid(pipeline.domain_gate(emb)).item()
            gate_pred = gate_prob > gate_threshold
            gate_correct += int(gate_pred)

            frame = pipeline.intent_extractor(emb)
            bridge_out = pipeline.bridge(frame)
            dec_out = pipeline.decoder(bridge_out)
            tier1_logits = dec_out["tier1_logits"]
            pred_idx = int(tier1_logits.argmax(dim=-1).item())
            pred_action = vocab.tier1_idx2token.get(pred_idx, "<UNK>")

            tier1_exact += int(pred_action == expected_action)
            domain_stats[domain]["total"] += 1
            action_stats[expected_action]["total"] += 1
            action_stats[expected_action]["correct"] += int(pred_action == expected_action)

            dsl = _render_predicted_dsl(pred_action, ex.shortcut_name)
            quality = _evaluate_dsl(dsl)
            parse_pass += int(quality["parsed"])
            validate_strict_pass += int(quality["validated_strict"])
            validate_perm_pass += int(quality["validated_permissive"])
            compile_strict_pass += int(quality["compiled_strict"])
            compile_perm_pass += int(quality["compiled_permissive"])
            runtime_unverified += int(quality["runtime_unverified"])
            domain_stats[domain]["compiled_strict"] += int(quality["compiled_strict"])

            if quality.get("error"):
                output_errors.append(
                    {
                        "shortcut_id": ex.shortcut_id,
                        "error": quality["error"],
                        "predicted_action": pred_action,
                    }
                )

            if fingerprint_enabled:
                logits_rows.append(tier1_logits.squeeze(0).cpu().numpy())
                action_predictions.append(pred_action)
                probe_labels.append(prompt[:80])

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{total} evaluated...")

    denom = total if total > 0 else 1
    results: dict[str, Any] = {
        "total_examples": total,
        "checkpoint": str(checkpoint_path),
        "eval_file": str(eval_file),
        "gate_accuracy": gate_correct / denom,
        "tier1_exact_match_rate": tier1_exact / denom,
        "parse_rate": parse_pass / denom,
        "validate_strict_rate": validate_strict_pass / denom,
        "validate_permissive_rate": validate_perm_pass / denom,
        "compile_strict_rate": compile_strict_pass / denom,
        "compile_permissive_rate": compile_perm_pass / denom,
        "runtime_unverified_compile_rate": runtime_unverified / denom,
        "domain_compile_strict_rate": {
            d: (v["compiled_strict"] / v["total"] if v["total"] else 0.0)
            for d, v in sorted(domain_stats.items())
        },
        "action_tier1_accuracy": {
            a: (v["correct"] / v["total"] if v["total"] else 0.0)
            for a, v in sorted(action_stats.items())
        },
        "num_errors": len(output_errors),
    }

    if output_errors:
        results["sample_errors"] = output_errors[:20]

    if fingerprint_enabled:
        _attach_fingerprint(
            results=results,
            checkpoint_meta=pipeline.checkpoint_meta,
            logits_rows=logits_rows,
            action_predictions=action_predictions,
            probe_labels=probe_labels,
            eval_cfg=eval_cfg,
            output_json=output_json,
        )

    if output_json:
        _write_results(results, output_json)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Balanced Sashimi checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval-file", type=Path, required=True, help="Path to typed_ir_eval.jsonl")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write evaluation results JSON",
    )
    args = parser.parse_args()

    config_data = yaml.safe_load(open(args.config))
    results = evaluate_checkpoint(
        config=config_data,
        checkpoint_path=args.checkpoint,
        eval_file=args.eval_file,
        output_json=args.output_json,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
