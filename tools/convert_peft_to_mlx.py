#!/usr/bin/env python3
"""
Convert a PEFT LoRA adapter (from Tinker/HuggingFace) to MLX-LM adapter format.

PEFT format:
  - adapter_config.json (PEFT config)
  - adapter_model.safetensors (keys: base_model.model.model.layers.X.module.lora_A.weight)

MLX-LM format:
  - adapter_config.json (MLX config with fine_tune_type, num_layers, lora_parameters)
  - adapters.safetensors (keys: layers.X.module.lora_a, weights transposed)

Usage:
    python scripts/convert_peft_to_mlx.py models/mini-v1 models/mini-v1-mlx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def peft_to_mlx_key(peft_key: str) -> str:
    """Convert a PEFT LoRA key to MLX-LM key format.

    PEFT keys: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    MLX keys:  model.layers.0.self_attn.q_proj.lora_a

    We strip 'base_model.model.' (keeping one 'model.' prefix) because
    MLX-LM model structure has model.layers.X... not layers.X...
    """
    key = peft_key.replace("base_model.model.", "", 1)
    key = key.replace(".lora_A.weight", ".lora_a")
    key = key.replace(".lora_B.weight", ".lora_b")
    return key


def convert_peft_to_mlx(
    peft_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Convert PEFT adapter to MLX-LM format.

    Args:
        peft_dir: Directory containing adapter_config.json and adapter_model.safetensors.
        output_dir: Directory to write adapters.safetensors and adapter_config.json.

    Returns:
        Dict with conversion statistics.
    """
    peft_dir = Path(peft_dir)
    output_dir = Path(output_dir)

    # Load PEFT config
    peft_config_path = peft_dir / "adapter_config.json"
    if not peft_config_path.exists():
        raise FileNotFoundError(f"PEFT config not found: {peft_config_path}")

    with open(peft_config_path) as f:
        peft_config = json.load(f)

    r = peft_config["r"]
    lora_alpha = peft_config.get("lora_alpha", r)
    lora_dropout = peft_config.get("lora_dropout", 0.0)
    target_modules = peft_config.get("target_modules", "all-linear")
    scale = lora_alpha / r

    print(f"PEFT config: r={r}, alpha={lora_alpha}, scale={scale:.2f}, dropout={lora_dropout}")
    print(f"Target modules: {target_modules}")

    # Load PEFT weights
    peft_weights_path = peft_dir / "adapter_model.safetensors"
    if not peft_weights_path.exists():
        raise FileNotFoundError(f"PEFT weights not found: {peft_weights_path}")

    mlx_weights = {}
    n_converted = 0
    layer_indices = set()

    with safe_open(str(peft_weights_path), framework="numpy") as f:
        for peft_key in f.keys():
            mlx_key = peft_to_mlx_key(peft_key)
            tensor = f.get_tensor(peft_key)

            # Transpose: PEFT stores (r, in) for lora_A, (out, r) for lora_B
            # MLX-LM expects (in, r) for lora_a, (r, out) for lora_b
            transposed = tensor.T
            mlx_weights[mlx_key] = mx.array(transposed)

            n_converted += 1

            # Track which layers have adapters
            # Keys are now model.layers.X.module.lora_a
            parts = mlx_key.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
                layer_indices.add(int(parts[2]))

    num_layers = len(layer_indices) if layer_indices else -1
    print(f"Converted {n_converted} weight tensors across {num_layers} layers")
    print(f"Layer range: {min(layer_indices)}-{max(layer_indices)}" if layer_indices else "")

    # Build MLX-LM config
    lora_params = {
        "rank": r,
        "dropout": lora_dropout,
        "scale": scale,
    }

    # If target_modules is a list of specific modules, include as keys
    if isinstance(target_modules, list):
        lora_params["keys"] = target_modules

    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_params,
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Wrote config: {config_path}")

    weights_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(weights_path), mlx_weights)
    print(f"Wrote weights: {weights_path}")

    return {
        "n_converted": n_converted,
        "num_layers": num_layers,
        "scale": scale,
        "config_path": str(config_path),
        "weights_path": str(weights_path),
    }


def main():
    parser = argparse.ArgumentParser(
        prog="convert_peft_to_mlx",
        description="Convert PEFT LoRA adapter to MLX-LM format",
    )
    parser.add_argument(
        "peft_dir",
        help="Directory containing PEFT adapter_config.json and adapter_model.safetensors",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for MLX-LM adapter files",
    )
    args = parser.parse_args()

    try:
        stats = convert_peft_to_mlx(args.peft_dir, args.output_dir)
        print(f"\nConversion complete: {stats['n_converted']} tensors → {stats['weights_path']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
