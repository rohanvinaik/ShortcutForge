"""Tests for Phase A cross-architecture study infrastructure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.behavioral_fingerprint import BehavioralFingerprint
from src.fingerprint_taxonomy import (
    TaxonomyReport,
    _js_divergence,
    build_taxonomy,
    cluster_fingerprints,
    compute_distance_matrix,
    fingerprint_to_vector,
)
from src.logit_normalizer import ActionVocab, extract_first_action, normalize_predictions
from src.model_registry import (
    ARCHITECTURE_FAMILIES,
    REGISTRY,
    ModelSpec,
    get_available,
    get_by_family,
    get_model_spec,
    load_yaml_overrides,
)


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_get_model_spec_exists(self):
        spec = get_model_spec("phi-2")
        assert spec.model_id == "phi-2"
        assert spec.display_name == "Phi-2"
        assert spec.architecture_family == "dense"
        assert spec.param_count_b == 2.7

    def test_get_model_spec_missing(self):
        with pytest.raises(KeyError, match="Unknown model_id"):
            get_model_spec("nonexistent-model")

    def test_get_by_family(self):
        swa_models = get_by_family("mistral_swa")
        ids = {s.model_id for s in swa_models}
        assert "cerebrum-7b" in ids
        assert "mistral-7b" in ids

    def test_get_by_family_empty(self):
        result = get_by_family("nonexistent_family")
        assert result == []

    def test_get_available_excludes_needs_download(self):
        available = get_available()
        available_ids = {s.model_id for s in available}
        # mamba-2.8b needs download and has no local_path
        assert "mamba-2.8b" not in available_ids
        # phi-2 does not need download
        assert "phi-2" in available_ids

    def test_registry_has_expected_count(self):
        assert len(REGISTRY) == 16

    def test_all_families_in_constant(self):
        families_in_registry = {s.architecture_family for s in REGISTRY.values()}
        for fam in families_in_registry:
            assert fam in ARCHITECTURE_FAMILIES

    def test_load_yaml_overrides(self, tmp_path):
        yaml_content = "models:\n  phi-2:\n    local_path: /tmp/phi2\n    enabled: false\n"
        yaml_path = tmp_path / "overrides.yaml"
        yaml_path.write_text(yaml_content)

        # Save originals
        orig_path = REGISTRY["phi-2"].local_path
        orig_enabled = REGISTRY["phi-2"].enabled

        try:
            load_yaml_overrides(yaml_path)
            assert REGISTRY["phi-2"].local_path == "/tmp/phi2"
            assert REGISTRY["phi-2"].enabled is False
        finally:
            # Restore originals
            REGISTRY["phi-2"].local_path = orig_path
            REGISTRY["phi-2"].enabled = orig_enabled


# ---------------------------------------------------------------------------
# Model Adapter
# ---------------------------------------------------------------------------


class TestModelAdapter:
    def test_load_adapter_standard(self):
        from src.model_adapter import HFModelAdapter, load_adapter

        spec = ModelSpec(
            model_id="test-std",
            display_name="Test Standard",
            hf_repo_or_path="test/model",
            architecture_family="dense",
            attention_type="dense",
            param_count_b=1.0,
            trust_remote_code=False,
        )
        adapter = load_adapter(spec)
        assert isinstance(adapter, HFModelAdapter)
        assert not adapter.is_loaded

    def test_load_adapter_exotic(self):
        from src.model_adapter import ExoticHFAdapter, load_adapter

        spec = ModelSpec(
            model_id="test-exotic",
            display_name="Test Exotic",
            hf_repo_or_path="test/exotic",
            architecture_family="ssm",
            attention_type="state_space",
            param_count_b=2.0,
            trust_remote_code=True,
        )
        adapter = load_adapter(spec)
        assert isinstance(adapter, ExoticHFAdapter)

    def test_get_metadata(self):
        from src.model_adapter import load_adapter

        spec = ModelSpec(
            model_id="test-meta",
            display_name="Test Meta",
            hf_repo_or_path="test/meta",
            architecture_family="dense",
            attention_type="dense",
            param_count_b=1.0,
        )
        adapter = load_adapter(spec)
        assert adapter.get_metadata() is spec

    def test_generate_not_loaded_raises(self):
        from src.model_adapter import HFModelAdapter

        spec = ModelSpec(
            model_id="test-err",
            display_name="Test",
            hf_repo_or_path="test/err",
            architecture_family="dense",
            attention_type="dense",
            param_count_b=1.0,
        )
        adapter = HFModelAdapter(spec)
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate(["hello"])

    def test_load_and_generate_with_mock(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Trigger the lazy backend check
            AutoModelForCausalLM.from_pretrained  # noqa: B018
        except (ImportError, AttributeError):
            pytest.skip("transformers with working PyTorch backend required")

        import torch

        from src.model_adapter import HFModelAdapter

        spec = ModelSpec(
            model_id="test-mock",
            display_name="Test Mock",
            hf_repo_or_path="test/mock",
            architecture_family="dense",
            attention_type="dense",
            param_count_b=1.0,
        )
        adapter = HFModelAdapter(spec)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.batch_decode.return_value = ["SHORTCUT\nACTION test\nENDSHORTCUT"]

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_config = MagicMock()
        mock_config.vocab_size = 100
        mock_model.config = mock_config
        mock_model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        mock_inputs = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer.return_value = mock_inputs

        with (
            patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tokenizer),
            patch.object(AutoModelForCausalLM, "from_pretrained", return_value=mock_model),
        ):
            adapter.load()
            assert adapter.is_loaded

            result = adapter.generate(["test prompt"])
            assert len(result) == 1

            adapter.unload()
            assert not adapter.is_loaded


# ---------------------------------------------------------------------------
# Logit Normalizer
# ---------------------------------------------------------------------------


class TestLogitNormalizer:
    def test_extract_first_action_basic(self):
        text = "SHORTCUT\nACTION is.workflow.actions.showalert\nENDSHORTCUT"
        assert extract_first_action(text) == "is.workflow.actions.showalert"

    def test_extract_first_action_multiline(self):
        text = (
            "SHORTCUT\n"
            "COMMENT Set a timer\n"
            "ACTION is.workflow.actions.settimer\n"
            "PARAMETER duration 5\n"
            "ACTION is.workflow.actions.showalert\n"
            "ENDSHORTCUT"
        )
        # Should return the FIRST action
        assert extract_first_action(text) == "is.workflow.actions.settimer"

    def test_extract_first_action_no_action(self):
        text = "This is just some text with no actions"
        assert extract_first_action(text) is None

    def test_extract_first_action_empty(self):
        assert extract_first_action("") is None

    def test_normalize_predictions_known(self):
        vocab = MagicMock(spec=ActionVocab)
        vocab.resolve.side_effect = lambda name: name if name.startswith("is.workflow") else None

        predictions = [
            "SHORTCUT\nACTION is.workflow.actions.showalert\nENDSHORTCUT",
            "SHORTCUT\nACTION unknown.action.name\nENDSHORTCUT",
            "No action here",
        ]
        result = normalize_predictions(predictions, vocab)
        assert result == ["is.workflow.actions.showalert", "<UNKNOWN>", "<UNKNOWN>"]

    @pytest.mark.skipif(
        not Path(
            Path(__file__).resolve().parent.parent.parent / "references" / "action_catalog.json"
        ).exists(),
        reason="action_catalog.json not found",
    )
    def test_action_vocab_loads(self):
        vocab = ActionVocab()
        assert len(vocab.actions) > 0
        assert len(vocab.aliases) > 0

    @pytest.mark.skipif(
        not Path(
            Path(__file__).resolve().parent.parent.parent / "references" / "action_catalog.json"
        ).exists(),
        reason="action_catalog.json not found",
    )
    def test_action_vocab_resolve(self):
        vocab = ActionVocab()
        # Exact match should work
        assert vocab.resolve("is.workflow.actions.alert") == "is.workflow.actions.alert"
        # Nonexistent should return None
        assert vocab.resolve("totally.fake.action.xyz") is None


# ---------------------------------------------------------------------------
# Fingerprint Taxonomy
# ---------------------------------------------------------------------------


class TestFingerprintTaxonomy:
    def test_fingerprint_to_vector_length(self):
        fp = BehavioralFingerprint(
            action_entropy=1.5,
            discreteness_score=0.7,
            variance_eigenvalues=[5.0, 3.0, 1.0],
            action_distribution={"a": 0.6, "b": 0.3, "c": 0.1},
        )
        vec = fingerprint_to_vector(fp)
        assert vec.shape == (27,)
        assert vec[0] == pytest.approx(1.5)
        assert vec[1] == pytest.approx(0.7)

    def test_fingerprint_to_vector_padding(self):
        fp = BehavioralFingerprint()
        vec = fingerprint_to_vector(fp)
        assert vec.shape == (27,)
        # All zeros for empty fingerprint
        assert np.all(vec == 0.0)

    def test_js_divergence_identical(self):
        dist = {"a": 0.5, "b": 0.5}
        assert _js_divergence(dist, dist) == pytest.approx(0.0, abs=1e-6)

    def test_js_divergence_different(self):
        dist_a = {"a": 1.0}
        dist_b = {"b": 1.0}
        js = _js_divergence(dist_a, dist_b)
        assert js > 0
        assert js <= 1.0  # JS divergence is bounded by log2(2) = 1

    def test_compute_distance_matrix_shape(self):
        fps = [
            BehavioralFingerprint(action_distribution={"a": 0.7, "b": 0.3}),
            BehavioralFingerprint(action_distribution={"a": 0.3, "b": 0.7}),
            BehavioralFingerprint(action_distribution={"c": 1.0}),
        ]
        ids = ["m1", "m2", "m3"]
        mat, ordered = compute_distance_matrix(fps, ids)
        assert mat.shape == (3, 3)
        assert ordered == ids
        # Diagonal should be zero
        assert mat[0, 0] == pytest.approx(0.0)
        # Symmetric
        assert mat[0, 1] == pytest.approx(mat[1, 0])

    @pytest.mark.skipif(not _can_import("scipy"), reason="scipy not installed")
    def test_cluster_fingerprints_basic(self):
        # Create two clear clusters
        n = 6
        dist = np.zeros((n, n))
        # Cluster 1: indices 0,1,2 are close to each other
        # Cluster 2: indices 3,4,5 are close to each other
        for i in range(3):
            for j in range(3):
                if i != j:
                    dist[i, j] = 0.1
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    dist[i, j] = 0.1
        # Between clusters: far apart
        for i in range(3):
            for j in range(3, 6):
                dist[i, j] = 0.9
                dist[j, i] = 0.9

        ids = [f"m{i}" for i in range(n)]
        result = cluster_fingerprints(dist, ids, n_clusters=2)
        assert result["n_clusters"] == 2
        assert len(result["labels"]) == n
        # First 3 should be in same cluster, last 3 in another
        assert result["labels"][0] == result["labels"][1] == result["labels"][2]
        assert result["labels"][3] == result["labels"][4] == result["labels"][5]
        assert result["labels"][0] != result["labels"][3]

    @pytest.mark.skipif(
        not (_can_import("scipy") and _can_import("sklearn")),
        reason="scipy/sklearn not installed",
    )
    def test_build_taxonomy_synthetic(self):
        # Two architecture families with distinct behaviors
        specs = [
            ModelSpec(
                model_id=f"dense-{i}",
                display_name=f"Dense {i}",
                hf_repo_or_path="x",
                architecture_family="dense",
                attention_type="dense",
                param_count_b=1.0,
            )
            for i in range(3)
        ] + [
            ModelSpec(
                model_id=f"ssm-{i}",
                display_name=f"SSM {i}",
                hf_repo_or_path="x",
                architecture_family="ssm",
                attention_type="state_space",
                param_count_b=2.0,
            )
            for i in range(3)
        ]

        # Dense models: concentrated on action "a"
        dense_fps = [
            BehavioralFingerprint(
                experiment_id=f"dense-{i}",
                action_entropy=0.5 + np.random.default_rng(i).random() * 0.1,
                action_distribution={"a": 0.8, "b": 0.2},
                discreteness_score=0.9,
                variance_eigenvalues=[5.0, 1.0],
            )
            for i in range(3)
        ]
        # SSM models: distributed across actions
        ssm_fps = [
            BehavioralFingerprint(
                experiment_id=f"ssm-{i}",
                action_entropy=2.0 + np.random.default_rng(i + 100).random() * 0.1,
                action_distribution={"c": 0.4, "d": 0.3, "e": 0.3},
                discreteness_score=0.3,
                variance_eigenvalues=[2.0, 2.0],
            )
            for i in range(3)
        ]

        fps = dense_fps + ssm_fps
        report = build_taxonomy(fps, specs)

        assert isinstance(report, TaxonomyReport)
        assert len(report.model_ids) == 6
        assert report.silhouette_score > 0

    @pytest.mark.skipif(
        not (_can_import("scipy") and _can_import("sklearn")),
        reason="scipy/sklearn not installed",
    )
    def test_taxonomy_report_serialization(self):
        report = TaxonomyReport(
            clusters={0: ["m1", "m2"], 1: ["m3"]},
            architecture_correlation=0.8,
            size_correlation=0.3,
            silhouette_score=0.6,
            model_ids=["m1", "m2", "m3"],
        )
        d = report.to_dict()
        assert d["architecture_correlation"] == 0.8
        assert "m1" in d["clusters"][0]

        md = report.to_markdown()
        assert "Behavioral Taxonomy Report" in md
        assert "Cluster 0" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
