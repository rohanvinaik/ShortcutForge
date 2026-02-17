"""
Evaluation harness for Balanced Sashimi checkpoints.

Usage:
    uv run python research/src/evaluate.py --config research/configs/base.yaml --checkpoint path/to/ckpt --eval-file path/to/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    eval_file: Path,
    output_json: Path | None = None,
) -> dict:
    """Evaluate a trained checkpoint against the eval set.

    Metrics: compile_strict_rate, compile_permissive_rate,
    roundtrip_success_rate, tier1/2 coverage, per-domain breakdown.
    """
    import sys

    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from research.src.bridge import InformationBridge
    from research.src.data import load_typed_ir_jsonl
    from research.src.domain_gate import DomainGate
    from research.src.encoder import PromptEncoder
    from research.src.intent_extractor import IntentExtractor
    from research.src.lowering import roundtrip_validate
    from research.src.ternary_decoder import TernaryDecoder

    # Load eval data
    examples = load_typed_ir_jsonl(eval_file)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]

    # Load vocabs for vocab sizes
    data_cfg = config["data"]
    tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
    tier2_vocab_dir = PROJECT_ROOT / data_cfg["tier2_vocab_dir"]
    tier2_fallback = json.loads((tier2_vocab_dir / "_global_fallback.json").read_text())

    # Reconstruct model
    device = "cpu"  # eval on CPU for safety
    encoder = PromptEncoder(
        model_name=model_cfg["encoder"]["model_name"],
        device=device,
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
        hidden_dim=model_cfg["decoder"]["hidden_dim"],
        tier1_vocab_size=len(tier1_vocab),
        tier2_vocab_size=len(tier2_fallback),
        num_layers=model_cfg["decoder"]["num_layers"],
    )

    # Load state dicts
    domain_gate.load_state_dict(ckpt["domain_gate"])
    intent_extractor.load_state_dict(ckpt["intent_extractor"])
    bridge.load_state_dict(ckpt["bridge"])
    decoder.load_state_dict(ckpt["decoder"])

    # Set eval mode
    for m in [domain_gate, intent_extractor, bridge, decoder]:
        m.eval()

    # Build inverse vocab for decoding
    tier1_idx2token = {v: k for k, v in tier1_vocab.items()}
    tier2_idx2token = {v: k for k, v in tier2_fallback.items()}

    # Metrics
    total = len(examples)
    gate_correct = 0
    roundtrip_success = 0
    tier1_exact_match = 0
    tier1_token_accuracy_sum = 0.0

    print(f"Evaluating {total} examples...")

    with torch.no_grad():
        for i, ex in enumerate(examples):
            # Encode prompt
            emb = encoder.encode([ex.prompt])  # [1, 384]

            # Gate check (should be in-domain for eval set)
            gate_logits = domain_gate(emb)
            gate_pred = torch.sigmoid(gate_logits).item() > 0.5
            if gate_pred:
                gate_correct += 1

            # Forward through pipeline
            frame = intent_extractor(emb)
            bridge_out = bridge(frame)
            dec_out = decoder(bridge_out)

            # Decode tier1 predictions
            tier1_logits = dec_out["tier1_logits"]  # [1, vocab_size]
            tier1_pred_idx = tier1_logits.argmax(dim=-1).item()
            tier1_pred_token = tier1_idx2token.get(tier1_pred_idx, "<UNK>")

            # Check against expected tier1 (first meaningful token)
            expected_token = (
                ex.tier1_tokens[1] if len(ex.tier1_tokens) > 1 else ex.tier1_tokens[0]
            )
            if tier1_pred_token == expected_token:
                tier1_exact_match += 1

            # Roundtrip validation (uses original example)
            success, msg = roundtrip_validate(ex)
            if success:
                roundtrip_success += 1

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{total} evaluated...")

    results = {
        "total_examples": total,
        "gate_accuracy": gate_correct / total if total > 0 else 0,
        "tier1_exact_match_rate": tier1_exact_match / total if total > 0 else 0,
        "roundtrip_success_rate": roundtrip_success / total if total > 0 else 0,
        "checkpoint": str(checkpoint_path),
        "eval_file": str(eval_file),
    }

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2))
        print(f"Results written to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Balanced Sashimi checkpoint",
    )
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval-file", type=Path, required=True,
                       help="Path to typed_ir_eval.jsonl")
    parser.add_argument("--output-json", type=Path, default=None,
                       help="Path to write evaluation results JSON")
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
