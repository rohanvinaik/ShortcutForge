"""Tests for Phase B: DSL evaluator, LoRA trainer, and cross-model comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.cross_model_comparison import (
    check_h1_architecture_signature,
    check_h5_scale_process_decoupling,
    compare_by_architecture,
    export_hypothesis_report,
)
from src.dsl_evaluator import (
    _extract_actions,
    _structural_check,
    compute_tier_accuracies,
    evaluate_outputs,
)
from src.pab_profile import PABCoreSeries, PABProfile, PABSummary, PABTierSeries

# ---------------------------------------------------------------------------
# DSL Evaluator tests
# ---------------------------------------------------------------------------


VALID_DSL = """SHORTCUT "Test"
ACTION is.workflow.actions.showresult
  text: "Hello"
ENDSHORTCUT"""

INVALID_DSL = """This is just some random text
without any DSL structure."""

PARTIAL_DSL = """SHORTCUT "Partial"
ACTION is.workflow.actions.showresult
  text: "Hello"
"""  # Missing ENDSHORTCUT


class TestExtractActions:
    def test_single_action(self):
        text = 'ACTION is.workflow.actions.showresult\n  text: "Hello"'
        assert _extract_actions(text) == ["is.workflow.actions.showresult"]

    def test_multiple_actions(self):
        text = (
            "ACTION is.workflow.actions.setvolume\n"
            "  level: 50\n"
            "ACTION is.workflow.actions.showresult\n"
            '  text: "Done"'
        )
        actions = _extract_actions(text)
        assert len(actions) == 2
        assert actions[0] == "is.workflow.actions.setvolume"
        assert actions[1] == "is.workflow.actions.showresult"

    def test_no_actions(self):
        assert _extract_actions("just some text") == []

    def test_indented_action(self):
        text = "  ACTION is.workflow.actions.delay\n    seconds: 5"
        assert _extract_actions(text) == ["is.workflow.actions.delay"]


class TestStructuralCheck:
    def test_valid_dsl(self):
        assert _structural_check(VALID_DSL) is True

    def test_invalid_dsl(self):
        assert _structural_check(INVALID_DSL) is False

    def test_partial_dsl_no_endshortcut(self):
        assert _structural_check(PARTIAL_DSL) is False

    def test_no_action(self):
        text = "SHORTCUT\nsome stuff\nENDSHORTCUT"
        assert _structural_check(text) is False


class TestEvaluateOutputs:
    def test_empty(self):
        metrics = evaluate_outputs([])
        assert metrics.endshortcut_rate == 0.0

    def test_all_valid(self):
        metrics = evaluate_outputs([VALID_DSL, VALID_DSL])
        assert metrics.endshortcut_rate == 1.0
        assert metrics.parse_rate == 1.0
        assert metrics.valid_action_rate == 1.0

    def test_mixed(self):
        metrics = evaluate_outputs([VALID_DSL, INVALID_DSL])
        assert metrics.endshortcut_rate == 0.5
        assert metrics.parse_rate == 0.5

    def test_endshortcut_rate(self):
        with_end = "SHORTCUT\nACTION test\nENDSHORTCUT"
        without_end = "SHORTCUT\nACTION test"
        metrics = evaluate_outputs([with_end, without_end])
        assert metrics.endshortcut_rate == 0.5

    def test_avg_action_count(self):
        two_actions = "ACTION a\nACTION b"
        one_action = "ACTION c"
        metrics = evaluate_outputs([two_actions, one_action])
        assert metrics.avg_action_count == 1.5

    def test_with_references(self):
        output = "SHORTCUT\nACTION is.workflow.actions.showresult\nENDSHORTCUT"
        ref = {
            "messages": [
                {"role": "user", "content": "show result"},
                {
                    "role": "assistant",
                    "content": "SHORTCUT\nACTION is.workflow.actions.showresult\nENDSHORTCUT",
                },
            ]
        }
        metrics = evaluate_outputs([output], [ref])
        assert metrics.first_action_accuracy == 1.0

    def test_with_reference_dsl_key(self):
        output = "SHORTCUT\nACTION is.workflow.actions.delay\nENDSHORTCUT"
        ref = {"reference_dsl": "SHORTCUT\nACTION is.workflow.actions.delay\nENDSHORTCUT"}
        metrics = evaluate_outputs([output], [ref])
        assert metrics.first_action_accuracy == 1.0


class TestTierAccuracies:
    def test_no_references(self):
        result = compute_tier_accuracies(["ACTION test"], None)
        assert result == {"tier1": 0.0, "tier2": 0.0}

    def test_tier1_match(self):
        output = "SHORTCUT\nACTION is.workflow.actions.showresult\n  text: hello\nENDSHORTCUT"
        ref = {
            "reference_dsl": (
                "SHORTCUT\nACTION is.workflow.actions.showresult\n  text: world\nENDSHORTCUT"
            ),
        }
        result = compute_tier_accuracies([output], [ref])
        assert result["tier1"] == 1.0
        # Same param key "text" -> tier2 match
        assert result["tier2"] == 1.0

    def test_tier1_mismatch(self):
        output = "ACTION is.workflow.actions.delay"
        ref = {"reference_dsl": "ACTION is.workflow.actions.showresult"}
        result = compute_tier_accuracies([output], [ref])
        assert result["tier1"] == 0.0

    def test_tier2_param_mismatch(self):
        output = "SHORTCUT\nACTION is.workflow.actions.showresult\n  text: hello\nENDSHORTCUT"
        ref = {
            "reference_dsl": (
                "SHORTCUT\nACTION is.workflow.actions.showresult\n  title: world\nENDSHORTCUT"
            ),
        }
        result = compute_tier_accuracies([output], [ref])
        assert result["tier1"] == 1.0
        assert result["tier2"] == 0.0


# ---------------------------------------------------------------------------
# LoRA Trainer tests
# ---------------------------------------------------------------------------


class TestLoRATrainerConfig:
    def test_config_defaults(self):
        from src.lora_trainer import LoRATrainConfig

        cfg = LoRATrainConfig(model_id="test-model")
        assert cfg.hparams.rank == 16
        assert cfg.hparams.alpha == 32
        assert cfg.hparams.lr == 2e-4
        assert cfg.hparams.max_steps == 1000
        assert cfg.schedule.checkpoint_interval == 50
        assert cfg.schedule.eval_interval == 100

    def test_config_custom(self):
        from src.lora_trainer import LoRATrainConfig, _LoRAHyperparams

        cfg = LoRATrainConfig(
            model_id="custom",
            hparams=_LoRAHyperparams(rank=8, alpha=16, lr=2e-4, batch_size=4, max_steps=500),
        )
        assert cfg.hparams.rank == 8
        assert cfg.hparams.max_steps == 500


class TestDetectTargetModules:
    def test_transformer_modules(self):
        """Detect q_proj/v_proj in a transformer-style model."""
        torch = pytest.importorskip("torch")

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 64)
                self.k_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.mlp = torch.nn.Linear(64, 128)

        from src.lora_trainer import LoRATrainer

        model = FakeTransformer()
        modules = LoRATrainer._detect_target_modules(model)
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_ssm_modules(self):
        """Detect in_proj/out_proj in a Mamba-style model."""
        torch = pytest.importorskip("torch")

        class FakeMamba(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = torch.nn.Linear(64, 128)
                self.out_proj = torch.nn.Linear(128, 64)
                self.norm = torch.nn.LayerNorm(64)

        from src.lora_trainer import LoRATrainer

        model = FakeMamba()
        modules = LoRATrainer._detect_target_modules(model)
        assert "in_proj" in modules
        assert "out_proj" in modules

    def test_fallback_linear(self):
        """Fall back to generic Linear layer detection."""
        torch = pytest.importorskip("torch")

        class FakeGeneric(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(64, 64)
                self.output_linear = torch.nn.Linear(64, 32)

        from src.lora_trainer import LoRATrainer

        model = FakeGeneric()
        modules = LoRATrainer._detect_target_modules(model)
        assert len(modules) > 0
        # "output_linear" contains "linear"
        assert "output_linear" in modules


class TestDataLoading:
    def test_load_jsonl(self, tmp_path):
        from src.lora_trainer import LoRATrainer

        data_file = tmp_path / "train.jsonl"
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "You generate DSL."},
                    {"role": "user", "content": "Set a timer"},
                    {"role": "assistant", "content": "SHORTCUT\nACTION timer\nENDSHORTCUT"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Toggle DND"},
                    {"role": "assistant", "content": "SHORTCUT\nACTION dnd\nENDSHORTCUT"},
                ]
            },
        ]
        with open(data_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        loaded = LoRATrainer._load_data(data_file)
        assert len(loaded) == 2
        assert "messages" in loaded[0]

    def test_load_missing_file(self, tmp_path):
        from src.lora_trainer import LoRATrainer

        loaded = LoRATrainer._load_data(tmp_path / "nonexistent.jsonl")
        assert loaded == []


# ---------------------------------------------------------------------------
# Cross-Model Comparison tests
# ---------------------------------------------------------------------------


def _make_profile(
    exp_id: str,
    stability_mean: float = 0.2,
    predictability: float = 0.05,
    tier1_accs: list[float] | None = None,
    crystallization_rate: float = 0.01,
    convergence_epoch: int | None = None,
    stability_regime: str = "moderate",
    repr_evolution: list[float] | None = None,
) -> PABProfile:
    """Build a minimal PABProfile for testing."""
    t1 = tier1_accs or [0.5, 0.7, 0.85]
    re = repr_evolution or [0.5, 0.3, 0.1]
    return PABProfile(
        experiment_id=exp_id,
        core=PABCoreSeries(
            stability=[stability_mean] * 3,
            representation_evolution=re,
        ),
        tiers=PABTierSeries(tier1_accuracy=t1),
        summary=PABSummary(
            stability_mean=stability_mean,
            predictability_final=predictability,
            crystallization_rate=crystallization_rate,
            convergence_epoch=convergence_epoch,
            stability_regime=stability_regime,
        ),
    )


@dataclass
class _FakeModelSpec:
    model_id: str = ""
    display_name: str = ""
    hf_repo_or_path: str = ""
    architecture_family: str = "dense"
    attention_type: str = "dense"
    param_count_b: float = 1.0
    local_path: str | None = None
    needs_download: bool = False
    chat_template: str | None = None
    trust_remote_code: bool = False
    memory_gb_estimate: float = 2.0


class TestArchitectureGrouping:
    def test_groups_by_family(self):
        profiles = [
            _make_profile("model-a"),
            _make_profile("model-b"),
            _make_profile("model-c"),
        ]
        specs = [
            _FakeModelSpec(model_id="model-a", architecture_family="dense"),
            _FakeModelSpec(model_id="model-b", architecture_family="ssm"),
            _FakeModelSpec(model_id="model-c", architecture_family="dense"),
        ]
        report = compare_by_architecture(profiles, specs)
        assert "dense" in report.architecture_groups
        assert "ssm" in report.architecture_groups
        assert len(report.architecture_groups["dense"]) == 2
        assert len(report.architecture_groups["ssm"]) == 1


class TestHypothesisH1:
    def test_insufficient_groups(self):
        profiles = [_make_profile("a")]
        arch_groups = {"dense": ["a"]}
        result = check_h1_architecture_signature(profiles, arch_groups)
        assert result["p_value"] == 1.0
        assert "insufficient" in result["interpretation"]

    def test_with_multiple_groups(self):
        profiles = [
            _make_profile("a", stability_mean=0.1),
            _make_profile("b", stability_mean=0.5),
        ]
        arch_groups = {"dense": ["a"], "ssm": ["b"]}
        result = check_h1_architecture_signature(profiles, arch_groups)
        # With only 1 per group, ANOVA may not be meaningful but should not crash
        assert "statistic" in result


class TestHypothesisH5:
    def test_known_correlation(self):
        """Synthetic profiles with known size-stability correlation."""
        profiles = [
            _make_profile("small", stability_mean=0.10),
            _make_profile("medium", stability_mean=0.20),
            _make_profile("large", stability_mean=0.30),
            _make_profile("xlarge", stability_mean=0.40),
        ]
        specs = [
            _FakeModelSpec(model_id="small", param_count_b=0.5),
            _FakeModelSpec(model_id="medium", param_count_b=1.0),
            _FakeModelSpec(model_id="large", param_count_b=3.0),
            _FakeModelSpec(model_id="xlarge", param_count_b=7.0),
        ]
        result = check_h5_scale_process_decoupling(profiles, specs)
        assert result["n_models"] == 4
        assert result["correlation"] > 0  # positive correlation: bigger = less stable
        assert result["r_squared"] > 0

    def test_insufficient_data(self):
        result = check_h5_scale_process_decoupling(
            [_make_profile("a")],
            [_FakeModelSpec(model_id="a")],
        )
        assert "insufficient" in result["interpretation"]


class TestExportHypothesisReport:
    def test_report_format(self):
        profiles = [_make_profile("a"), _make_profile("b")]
        specs = [
            _FakeModelSpec(model_id="a", architecture_family="dense"),
            _FakeModelSpec(model_id="b", architecture_family="ssm"),
        ]
        report = compare_by_architecture(profiles, specs)
        md = export_hypothesis_report(report)

        assert "Cross-Architecture Hypothesis Testing Report" in md
        assert "H1" in md
        assert "H5" in md
        assert "Architecture Groups" in md
        assert "dense" in md
        assert "ssm" in md


# ---------------------------------------------------------------------------
# LoRA Compatibility tests (architecture-specific target module detection)
# ---------------------------------------------------------------------------


class TestLoRACompatibility:
    def test_mixed_architecture(self):
        """Model with both transformer and SSM layers prefers transformer targets."""
        torch = pytest.importorskip("torch")

        class FakeHybrid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.in_proj = torch.nn.Linear(64, 128)
                self.out_proj = torch.nn.Linear(128, 64)

        from src.lora_trainer import LoRATrainer

        model = FakeHybrid()
        modules = LoRATrainer._detect_target_modules(model)
        # Should prefer transformer targets
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_no_linear_layers(self):
        """Model with no Linear layers falls back to defaults."""
        torch = pytest.importorskip("torch")

        class FakeEmpty(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)

        from src.lora_trainer import LoRATrainer

        model = FakeEmpty()
        modules = LoRATrainer._detect_target_modules(model)
        # Should return default fallback
        assert modules == ["q_proj", "v_proj"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
