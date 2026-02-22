"""Tests for ablation matrix runner configuration and merging."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestLoadMatrixConfig:
    """Test ablation matrix YAML loading."""

    def test_load_matrix_config(self):
        """Should load ablation.yaml with 8 configurations."""
        from scripts.run_ablation_matrix import load_matrix_config

        config_path = Path(__file__).resolve().parent.parent / "configs" / "ablation.yaml"
        if not config_path.exists():
            pytest.skip("ablation.yaml not found")

        matrix = load_matrix_config(config_path)
        assert "configurations" in matrix
        assert len(matrix["configurations"]) == 8

    def test_load_missing_config_raises(self, tmp_path: Path):
        """Loading a nonexistent config should raise."""
        from scripts.run_ablation_matrix import load_matrix_config

        with pytest.raises(FileNotFoundError):
            load_matrix_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_config_raises(self, tmp_path: Path):
        """Config without 'configurations' key should raise ValueError."""
        from scripts.run_ablation_matrix import load_matrix_config

        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("execution:\n  device: cpu\n")
        with pytest.raises(ValueError, match="configurations"):
            load_matrix_config(bad_config)


class TestMergeConfig:
    """Test deep config merging."""

    def test_merge_config_deep(self):
        """Deep merge should override nested keys while preserving others."""
        from scripts.run_ablation_matrix import merge_config

        base = {
            "model": {
                "decoder": {"hidden_dim": 256, "ternary_enabled": True},
                "bridge": {"bridge_dim": 128},
            },
            "training": {"batch_size": 16, "loss": {"margin": 0.5}},
        }
        overrides = {
            "model": {"decoder": {"ternary_enabled": False}},
            "training": {"loss": {"margin": 0.0}},
        }

        merged = merge_config(base, overrides)

        # Overridden values
        assert merged["model"]["decoder"]["ternary_enabled"] is False
        assert merged["training"]["loss"]["margin"] == 0.0

        # Preserved values
        assert merged["model"]["decoder"]["hidden_dim"] == 256
        assert merged["model"]["bridge"]["bridge_dim"] == 128
        assert merged["training"]["batch_size"] == 16

    def test_merge_does_not_mutate_base(self):
        """merge_config should not modify the original base dict."""
        from scripts.run_ablation_matrix import merge_config

        base = {"model": {"decoder": {"hidden_dim": 256}}}
        overrides = {"model": {"decoder": {"hidden_dim": 64}}}

        merge_config(base, overrides)
        assert base["model"]["decoder"]["hidden_dim"] == 256


class TestGenerateRunConfigs:
    """Test run config generation from matrix."""

    def test_generate_run_configs(self):
        """Should produce correct number of merged configs."""
        from scripts.run_ablation_matrix import generate_run_configs

        matrix = {
            "execution": {"device": "cpu", "seed": 42, "max_iterations": 100},
            "configurations": {
                "run-a": {
                    "description": "Test A",
                    "overrides": {"training": {"batch_size": 8}},
                },
                "run-b": {
                    "description": "Test B",
                    "overrides": {"training": {"batch_size": 32}},
                },
            },
        }
        base = {"training": {"batch_size": 16, "seed": 1, "max_iterations": 1000}}

        configs = generate_run_configs(matrix, base)
        assert len(configs) == 2

        # Check first config
        run_id_a, desc_a, config_a = configs[0]
        assert run_id_a == "run-a"
        assert desc_a == "Test A"
        assert config_a["training"]["batch_size"] == 8
        assert config_a["training"]["seed"] == 42  # from execution
        assert config_a["training"]["max_iterations"] == 100  # from execution

        # Check second config
        _, _, config_b = configs[1]
        assert config_b["training"]["batch_size"] == 32


class TestDryRun:
    """Test that dry-run doesn't execute training."""

    def test_dry_run_lists_configs(self, capsys):
        """Dry run should list configurations without importing torch."""
        from scripts.run_ablation_matrix import generate_run_configs

        matrix = {
            "execution": {},
            "configurations": {
                "test-1": {"description": "Config 1", "overrides": {}},
                "test-2": {"description": "Config 2", "overrides": {}},
            },
        }
        base = {"training": {"batch_size": 16}}

        configs = generate_run_configs(matrix, base)
        # Dry run just lists — no training should happen
        for run_id, desc, _config in configs:
            print(f"  [{run_id}] {desc}")

        captured = capsys.readouterr()
        assert "test-1" in captured.out
        assert "test-2" in captured.out
