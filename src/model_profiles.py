"""Model profile loader for ShortcutForge distillation pipeline.

Loads runtime policy profiles and promotion gates from
configs/model_profiles.yaml. Each profile defines a model + adapter +
chat template + validation/fallback/timeout policy.

Usage:
    from model_profiles import load_profiles, get_profile, get_promotion_gates

    profiles = load_profiles()
    p = get_profile("tiny_qwen_mainline")
    gates = get_promotion_gates()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "model_profiles.yaml"


@dataclass
class ModelProfile:
    """A model runtime policy profile."""
    name: str
    model_path: str
    adapter_path: str | None
    chat_template: str | None  # "llama3", "chatml", or None (API models)
    validator_mode: str = "strict_plus_permissive"
    dynamic_budget: bool = True
    grammar_retry_only: bool = True
    max_retries: int = 3
    fallback_order: list[str] = field(default_factory=list)
    timeout_s: float = 90.0

    def resolved_adapter_path(self) -> str | None:
        """Resolve adapter_path relative to project root if not absolute."""
        if self.adapter_path is None:
            return None
        p = Path(self.adapter_path)
        if p.is_absolute():
            return str(p)
        resolved = _PROJECT_ROOT / p
        return str(resolved) if resolved.exists() else self.adapter_path


@dataclass
class PromotionGate:
    """A single promotion gate threshold."""
    metric: str
    direction: str  # "min" or "max"
    threshold: float

    def passes(self, value: float) -> bool:
        """Check if the given value passes this gate."""
        if self.direction == "min":
            return value >= self.threshold
        return value <= self.threshold

    def __str__(self) -> str:
        op = ">=" if self.direction == "min" else "<="
        return f"{self.metric} {op} {self.threshold}"


@dataclass
class ProfileConfig:
    """Complete parsed configuration."""
    profiles: dict[str, ModelProfile]
    default_profile: str
    promotion_gates: list[PromotionGate]


def load_profiles(config_path: str | Path | None = None) -> ProfileConfig:
    """Load model profiles from YAML config.

    Args:
        config_path: Path to config file. Defaults to configs/model_profiles.yaml.

    Returns:
        ProfileConfig with all profiles and promotion gates.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "profiles" not in raw:
        raise ValueError(f"Invalid profile config: missing 'profiles' key in {path}")

    profiles: dict[str, ModelProfile] = {}
    for name, pdata in raw["profiles"].items():
        profiles[name] = ModelProfile(
            name=name,
            model_path=pdata["model_path"],
            adapter_path=pdata.get("adapter_path"),
            chat_template=pdata.get("chat_template"),
            validator_mode=pdata.get("validator_mode", "strict_plus_permissive"),
            dynamic_budget=pdata.get("dynamic_budget", True),
            grammar_retry_only=pdata.get("grammar_retry_only", True),
            max_retries=pdata.get("max_retries", 3),
            fallback_order=pdata.get("fallback_order", []),
            timeout_s=float(pdata.get("timeout_s", 90)),
        )

    default_name = raw.get("default_profile", "local_8b_mainline")
    if default_name not in profiles:
        raise ValueError(
            f"Default profile '{default_name}' not found in profiles: "
            f"{list(profiles.keys())}"
        )

    gates: list[PromotionGate] = []
    for metric, spec in raw.get("promotion_gates", {}).items():
        if "min" in spec:
            gates.append(PromotionGate(metric=metric, direction="min", threshold=float(spec["min"])))
        elif "max" in spec:
            gates.append(PromotionGate(metric=metric, direction="max", threshold=float(spec["max"])))

    return ProfileConfig(
        profiles=profiles,
        default_profile=default_name,
        promotion_gates=gates,
    )


def get_profile(name: str, config_path: str | Path | None = None) -> ModelProfile:
    """Get a single profile by name.

    Raises:
        KeyError: If profile name doesn't exist.
    """
    config = load_profiles(config_path)
    if name not in config.profiles:
        raise KeyError(
            f"Profile '{name}' not found. Available: {list(config.profiles.keys())}"
        )
    return config.profiles[name]


def get_default_profile(config_path: str | Path | None = None) -> ModelProfile:
    """Get the default profile."""
    config = load_profiles(config_path)
    return config.profiles[config.default_profile]


def get_promotion_gates(config_path: str | Path | None = None) -> list[PromotionGate]:
    """Get all promotion gates."""
    config = load_profiles(config_path)
    return config.promotion_gates


def check_promotion(
    metrics: dict[str, float],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check metrics against all promotion gates.

    Args:
        metrics: Dict mapping metric names to values.
        config_path: Optional config path override.

    Returns:
        Dict with 'passed' (bool), 'gates' (list of gate results).
    """
    gates = get_promotion_gates(config_path)
    gate_results = []
    all_passed = True

    for gate in gates:
        value = metrics.get(gate.metric)
        if value is None:
            gate_results.append({
                "gate": str(gate),
                "metric": gate.metric,
                "value": None,
                "passed": False,
                "reason": f"Metric '{gate.metric}' not found in results",
            })
            all_passed = False
        else:
            passed = gate.passes(value)
            gate_results.append({
                "gate": str(gate),
                "metric": gate.metric,
                "value": value,
                "threshold": gate.threshold,
                "passed": passed,
            })
            if not passed:
                all_passed = False

    return {
        "passed": all_passed,
        "gates": gate_results,
        "total_gates": len(gates),
        "gates_passed": sum(1 for g in gate_results if g["passed"]),
    }


def list_profiles(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """List all profiles as dicts for display/MCP.

    Returns:
        List of profile summary dicts.
    """
    config = load_profiles(config_path)
    result = []
    for name, p in config.profiles.items():
        result.append({
            "name": name,
            "model_path": p.model_path,
            "adapter_path": p.adapter_path,
            "chat_template": p.chat_template,
            "validator_mode": p.validator_mode,
            "max_retries": p.max_retries,
            "fallback_order": p.fallback_order,
            "timeout_s": p.timeout_s,
            "is_default": name == config.default_profile,
        })
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        config = load_profiles()
        print(f"Loaded {len(config.profiles)} profiles from {_DEFAULT_CONFIG}")
        print(f"Default: {config.default_profile}")
        print(f"Promotion gates: {len(config.promotion_gates)}")
        for g in config.promotion_gates:
            print(f"  {g}")
        print()
        for name, p in config.profiles.items():
            default_tag = " (DEFAULT)" if name == config.default_profile else ""
            print(f"  {name}{default_tag}: {p.model_path}")
            print(f"    adapter={p.adapter_path}, template={p.chat_template}")
            print(f"    fallback={p.fallback_order}, retries={p.max_retries}")
        print("\nAll OK.")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
