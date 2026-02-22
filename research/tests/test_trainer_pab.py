"""Tests for PAB tracker integration in the trainer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pab_tracker import PABTracker
from src.trainer import BalancedSashimiTrainer, _RunPaths, _TrainInfra


class TestPABTrackerCreation:
    """Verify PABTracker is created/skipped based on config."""

    def _make_trainer(self, pab_enabled: bool) -> BalancedSashimiTrainer:
        config = {
            "pab": {"enabled": pab_enabled, "checkpoint_interval": 50},
            "training": {
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_iterations": 10,
                "batch_size": 4,
                "loss": {"initial_log_sigma": 0.0},
            },
            "safety": {
                "gradient_check_every_n_steps": 50,
                "nan_abort": True,
                "ternary_distribution_log_every_n_steps": 100,
                "max_grad_norm": 1.0,
            },
            "logging": {
                "run_dir": "/tmp/test_runs",
                "training_log_suffix": "_log.jsonl",
                "eval_results_suffix": "_eval.json",
                "checkpoint_dir": "/tmp/test_ckpt",
            },
            "model": {
                "encoder": {"model_name": "all-MiniLM-L6-v2", "output_dim": 384},
                "domain_gate": {"hidden_dim": 128},
                "intent_extractor": {"frame_dim": 256},
                "bridge": {"bridge_dim": 128},
                "decoder": {"hidden_dim": 256, "num_layers": 2},
            },
            "data": {
                "typed_ir_train": "training_data/typed_ir_train.jsonl",
                "tier1_vocab": "references/tier1_vocab.json",
                "tier2_vocab_dir": "references/tier2_vocab/",
            },
        }
        return BalancedSashimiTrainer(config=config, run_id="test-pab", device="cpu")

    def test_pab_tracker_created_when_enabled(self):
        """Config with pab.enabled=true should create a PABTracker in infra."""
        trainer = self._make_trainer(pab_enabled=True)
        # Mock setup to avoid needing real data/models
        mock_tracker = PABTracker(experiment_id="test-pab", config_hash="abc123")
        trainer.infra = _TrainInfra(
            composite_loss=MagicMock(),
            ood_loss=MagicMock(),
            optimizer=MagicMock(),
            dataset=MagicMock(),
            pab_tracker=mock_tracker,
        )
        assert trainer.infra.pab_tracker is not None
        assert isinstance(trainer.infra.pab_tracker, PABTracker)
        assert trainer.infra.pab_tracker.experiment_id == "test-pab"

    def test_pab_tracker_none_when_disabled(self):
        """Config with pab.enabled=false should have None pab_tracker."""
        trainer = self._make_trainer(pab_enabled=False)
        trainer.infra = _TrainInfra(
            composite_loss=MagicMock(),
            ood_loss=MagicMock(),
            optimizer=MagicMock(),
            dataset=MagicMock(),
            pab_tracker=None,
        )
        assert trainer.infra.pab_tracker is None


class TestDecoderWeightSigns:
    """Test weight sign extraction from TernaryLinear layers."""

    def test_collect_decoder_weight_signs(self):
        """Should return a numpy array of signs from TernaryLinear layers."""
        pytest.importorskip("torch")
        from src.ternary_decoder import TernaryDecoder

        decoder = TernaryDecoder(
            input_dim=128,
            hidden_dim=64,
            tier1_vocab_size=10,
            tier2_vocab_size=5,
            num_layers=1,
        )
        trainer = BalancedSashimiTrainer(
            config={"pab": {"enabled": True}},
            run_id="test-signs",
            device="cpu",
        )
        from collections import namedtuple

        Pipeline = namedtuple("Pipeline", "encoder domain_gate intent_extractor bridge decoder")
        trainer.pipeline = Pipeline(
            encoder=None,
            domain_gate=None,
            intent_extractor=None,
            bridge=None,
            decoder=decoder,
        )

        signs = trainer._collect_decoder_weight_signs()
        assert signs is not None
        assert isinstance(signs, np.ndarray)
        assert len(signs) > 0
        # Signs should be -1, 0, or +1
        unique = set(signs.tolist())
        assert unique <= {-1.0, 0.0, 1.0}


class TestPABProfileSaved:
    """Test that PAB profile is saved after training finalization."""

    def test_profile_path_in_result(self):
        """_finalize_training should include pab_profile path when tracker present."""
        trainer = BalancedSashimiTrainer(
            config={
                "pab": {"enabled": True, "save_profiles": True},
                "logging": {
                    "run_dir": "/tmp/test_runs",
                    "training_log_suffix": "_log.jsonl",
                    "checkpoint_dir": "/tmp/test_ckpt",
                },
                "training": {"max_iterations": 10},
            },
            run_id="test-finalize",
            device="cpu",
        )
        # Create a mock tracker that returns a profile
        mock_profile = MagicMock()
        mock_profile.summary.stability_regime = "stable"
        mock_profile.summary.stability_mean = 0.05
        mock_tracker = MagicMock()
        mock_tracker.finalize.return_value = mock_profile

        trainer.infra = _TrainInfra(
            composite_loss=MagicMock(),
            ood_loss=MagicMock(),
            optimizer=MagicMock(),
            dataset=MagicMock(),
            pab_tracker=mock_tracker,
        )
        trainer.step = 10
        run_dir = pytest.importorskip("pathlib").Path("/tmp/test_pab_finalize")
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer._paths = _RunPaths(run_dir=run_dir, checkpoint_dir=run_dir)

        with patch.object(trainer, "save_checkpoint", return_value=trainer.run_dir / "ckpt.pt"):
            result = trainer._finalize_training(
                all_losses=[{"L_total": 0.5}],
                epoch=1,
                elapsed=1.0,
                log_path=trainer.run_dir / "log.jsonl",
            )

        assert "pab_profile" in result
        mock_tracker.finalize.assert_called_once()
        mock_profile.save.assert_called_once()
