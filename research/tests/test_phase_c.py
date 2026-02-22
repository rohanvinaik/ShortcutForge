"""Tests for Phase C encoder adapter and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _MockModelSpec:
    """Minimal ModelSpec stand-in for tests."""

    model_id: str = "test-model"
    display_name: str = "Test Model"
    hf_repo_or_path: str = "test/model"
    architecture_family: str = "dense"
    attention_type: str = "mha"
    param_count_b: float = 0.5
    local_path: str | None = None
    needs_download: bool = False
    chat_template: str | None = None
    trust_remote_code: bool = False
    memory_gb_estimate: float = 1.0


class TestExternalEncoder:
    def test_init(self):
        from src.encoder_adapter import ExternalEncoder

        spec = _MockModelSpec()
        enc = ExternalEncoder(model_spec=spec, target_dim=384)
        assert enc.model_spec is spec
        assert enc.target_dim == 384
        assert enc._adapter is None
        assert enc._projection is None
        assert enc._hidden_dim is None

    def test_init_custom_dim(self):
        from src.encoder_adapter import ExternalEncoder

        enc = ExternalEncoder(model_spec=_MockModelSpec(), target_dim=256)
        assert enc.target_dim == 256

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch required"),
        reason="torch not available",
    )
    def test_output_shape(self):
        import torch

        from src.encoder_adapter import ExternalEncoder

        spec = _MockModelSpec()
        enc = ExternalEncoder(model_spec=spec, target_dim=384)

        # Manually wire up mock internals
        hidden_dim = 768
        enc._hidden_dim = hidden_dim
        enc._device = "cpu"
        enc._projection = torch.nn.Linear(hidden_dim, 384, bias=False)

        # Mock _extract_hidden_states to return a known tensor
        mock_hidden = torch.randn(2, hidden_dim)
        enc._extract_hidden_states = MagicMock(return_value=mock_hidden)

        result = enc.encode(["hello", "world"])
        assert result.shape == (2, 384)
        enc._extract_hidden_states.assert_called_once_with(["hello", "world"])

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch required"),
        reason="torch not available",
    )
    def test_forward_aliases_encode(self):
        import torch

        from src.encoder_adapter import ExternalEncoder

        spec = _MockModelSpec()
        enc = ExternalEncoder(model_spec=spec, target_dim=128)
        enc._hidden_dim = 512
        enc._device = "cpu"
        enc._projection = torch.nn.Linear(512, 128, bias=False)
        enc._extract_hidden_states = MagicMock(return_value=torch.randn(1, 512))

        result = enc.forward(["test"])
        assert result.shape == (1, 128)

    def test_unload(self):
        from src.encoder_adapter import ExternalEncoder

        spec = _MockModelSpec()
        enc = ExternalEncoder(model_spec=spec)
        mock_adapter = MagicMock()
        enc._adapter = mock_adapter
        enc._projection = MagicMock()
        enc._hidden_dim = 768

        enc.unload()

        mock_adapter.unload.assert_called_once()
        assert enc._adapter is None
        assert enc._projection is None
        assert enc._hidden_dim is None

    def test_unload_when_already_unloaded(self):
        from src.encoder_adapter import ExternalEncoder

        enc = ExternalEncoder(model_spec=_MockModelSpec())
        enc.unload()  # should not raise


class TestBuildEncoder:
    @patch("src.model_adapter.load_adapter")
    def test_factory(self, mock_load_adapter):
        torch = pytest.importorskip("torch")
        from src.encoder_adapter import ExternalEncoder, build_encoder

        # Mock the adapter and its model
        mock_adapter = MagicMock()
        mock_load_adapter.return_value = mock_adapter

        # Mock the model forward pass for hidden dim probing
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter._model = mock_model
        mock_adapter._tokenizer = mock_tokenizer

        # Tokenizer returns mock inputs
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        # Model returns mock hidden states
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(1, 5, 512)]
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = iter([torch.randn(2, 2)])

        spec = _MockModelSpec()
        encoder = build_encoder(spec, target_dim=384, device="cpu")

        assert isinstance(encoder, ExternalEncoder)
        assert encoder._hidden_dim == 512
        assert encoder._projection is not None
        mock_load_adapter.assert_called_once_with(spec)
        mock_adapter.load.assert_called_once()


class TestEncoderSwap:
    def test_trainer_accepts_override(self):
        from src.trainer import BalancedSashimiTrainer

        mock_encoder = MagicMock()
        trainer = BalancedSashimiTrainer(
            config={"training": {"max_iterations": 10}},
            run_id="test",
            encoder_override=mock_encoder,
        )
        assert trainer.encoder_override is mock_encoder

    def test_trainer_default_no_override(self):
        from src.trainer import BalancedSashimiTrainer

        trainer = BalancedSashimiTrainer(
            config={"training": {"max_iterations": 10}},
            run_id="test",
        )
        assert trainer.encoder_override is None


class TestPhaseCAblationMatrix:
    def test_config_loading(self, tmp_path):
        config_content = {
            "base_config": "configs/base.yaml",
            "selected_encoders": [
                {"model_id": "test-model", "description": "Test"},
            ],
            "ablation_configs": {
                "C.1": {
                    "description": "Baseline",
                    "overrides": {"model": {"decoder": {"ternary_enabled": False}}},
                },
            },
            "execution": {"device": "cpu", "seed": 42, "max_iterations": 100},
        }

        import yaml

        config_path = tmp_path / "phase_c.yaml"
        config_path.write_text(yaml.dump(config_content))

        from scripts.run_phase_c import load_phase_c_config

        loaded = load_phase_c_config(config_path)
        assert len(loaded["selected_encoders"]) == 1
        assert "C.1" in loaded["ablation_configs"]

    def test_config_missing_encoders(self, tmp_path):
        import yaml

        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump({"ablation_configs": {}}))

        from scripts.run_phase_c import load_phase_c_config

        with pytest.raises(ValueError, match="selected_encoders"):
            load_phase_c_config(config_path)

    def test_config_missing_ablations(self, tmp_path):
        import yaml

        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump({"selected_encoders": []}))

        from scripts.run_phase_c import load_phase_c_config

        with pytest.raises(ValueError, match="ablation_configs"):
            load_phase_c_config(config_path)

    def test_run_id_format(self):
        from scripts.run_phase_c import build_run_matrix

        phase_config = {
            "selected_encoders": [
                {"model_id": "qwen2.5-0.5b"},
                {"model_id": "phi-2"},
            ],
            "ablation_configs": {
                "C.1": {"description": "Baseline", "overrides": {}},
                "C.2": {"description": "Negatives", "overrides": {}},
            },
            "execution": {},
        }
        base_config = {"model": {}, "training": {}}

        runs = build_run_matrix(phase_config, base_config)

        assert len(runs) == 4  # 2 encoders x 2 ablations
        run_ids = [r[0] for r in runs]
        assert "qwen2.5-0.5b_C.1" in run_ids
        assert "qwen2.5-0.5b_C.2" in run_ids
        assert "phi-2_C.1" in run_ids
        assert "phi-2_C.2" in run_ids

    def test_encoder_filter(self):
        from scripts.run_phase_c import build_run_matrix

        phase_config = {
            "selected_encoders": [
                {"model_id": "qwen2.5-0.5b"},
                {"model_id": "phi-2"},
            ],
            "ablation_configs": {
                "C.1": {"description": "Baseline", "overrides": {}},
            },
            "execution": {},
        }
        base_config = {"model": {}, "training": {}}

        runs = build_run_matrix(phase_config, base_config, encoder_filter=["phi-2"])

        assert len(runs) == 1
        assert runs[0][0] == "phi-2_C.1"

    def test_merge_config(self):
        from scripts.run_phase_c import merge_config

        base = {
            "model": {"encoder": {"dim": 384}, "decoder": {"layers": 2}},
            "training": {"lr": 1e-4},
        }
        overrides = {"model": {"decoder": {"layers": 4}}, "training": {"lr": 1e-3}}

        merged = merge_config(base, overrides)

        assert merged["model"]["encoder"]["dim"] == 384  # untouched
        assert merged["model"]["decoder"]["layers"] == 4  # overridden
        assert merged["training"]["lr"] == 1e-3  # overridden
        # Original not mutated
        assert base["model"]["decoder"]["layers"] == 2

    def test_execution_overrides_applied(self):
        from scripts.run_phase_c import build_run_matrix

        phase_config = {
            "selected_encoders": [{"model_id": "test"}],
            "ablation_configs": {"C.1": {"overrides": {}}},
            "execution": {"seed": 123, "max_iterations": 500},
        }
        base_config = {"training": {"seed": 42, "max_iterations": 1000}}

        runs = build_run_matrix(phase_config, base_config)
        _, _, _, config = runs[0]

        assert config["training"]["seed"] == 123
        assert config["training"]["max_iterations"] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
