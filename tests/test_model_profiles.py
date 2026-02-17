"""Tests for model_profiles.py — profile loader and promotion gates."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from model_profiles import (
    ModelProfile,
    ProfileConfig,
    PromotionGate,
    check_promotion,
    get_default_profile,
    get_profile,
    get_promotion_gates,
    list_profiles,
    load_profiles,
)

# Minimal valid config for testing
_TEST_CONFIG = {
    "profiles": {
        "test_profile_a": {
            "model_path": "test/model-a",
            "adapter_path": "models/test-adapter",
            "chat_template": "llama3",
            "validator_mode": "strict_plus_permissive",
            "dynamic_budget": True,
            "grammar_retry_only": True,
            "max_retries": 3,
            "fallback_order": ["claude"],
            "timeout_s": 60,
        },
        "test_profile_b": {
            "model_path": "test/model-b",
            "adapter_path": None,
            "chat_template": "chatml",
            "max_retries": 2,
            "fallback_order": ["test_profile_a", "claude"],
            "timeout_s": 30,
        },
    },
    "default_profile": "test_profile_a",
    "promotion_gates": {
        "compile_strict_rate": {"min": 95.0},
        "compile_permissive_rate": {"min": 97.0},
        "runtime_unverified_compile_rate": {"max": 2.0},
        "fallback_rate": {"max": 5.0},
        "health_logger_scenario_score": {"min": 0.85},
    },
}


def _write_test_config(config: dict) -> str:
    """Write config to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f)
    return path


class TestLoadProfiles(unittest.TestCase):
    """Test profile loading from YAML."""

    def setUp(self):
        self.config_path = _write_test_config(_TEST_CONFIG)

    def tearDown(self):
        os.unlink(self.config_path)

    def test_load_valid_config(self):
        config = load_profiles(self.config_path)
        self.assertIsInstance(config, ProfileConfig)
        self.assertEqual(len(config.profiles), 2)
        self.assertEqual(config.default_profile, "test_profile_a")

    def test_profile_fields(self):
        config = load_profiles(self.config_path)
        p = config.profiles["test_profile_a"]
        self.assertEqual(p.model_path, "test/model-a")
        self.assertEqual(p.adapter_path, "models/test-adapter")
        self.assertEqual(p.chat_template, "llama3")
        self.assertEqual(p.max_retries, 3)
        self.assertEqual(p.fallback_order, ["claude"])
        self.assertEqual(p.timeout_s, 60.0)

    def test_profile_defaults(self):
        config = load_profiles(self.config_path)
        p = config.profiles["test_profile_b"]
        self.assertEqual(p.validator_mode, "strict_plus_permissive")  # default
        self.assertTrue(p.dynamic_budget)  # default
        self.assertTrue(p.grammar_retry_only)  # default

    def test_chatml_template(self):
        config = load_profiles(self.config_path)
        p = config.profiles["test_profile_b"]
        self.assertEqual(p.chat_template, "chatml")

    def test_null_adapter(self):
        config = load_profiles(self.config_path)
        p = config.profiles["test_profile_b"]
        self.assertIsNone(p.adapter_path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_profiles("/nonexistent/path.yaml")

    def test_malformed_config(self):
        path = _write_test_config({"not_profiles": {}})
        try:
            with self.assertRaises(ValueError):
                load_profiles(path)
        finally:
            os.unlink(path)

    def test_invalid_default_profile(self):
        bad_config = dict(_TEST_CONFIG)
        bad_config["default_profile"] = "nonexistent"
        path = _write_test_config(bad_config)
        try:
            with self.assertRaises(ValueError):
                load_profiles(path)
        finally:
            os.unlink(path)


class TestGetProfile(unittest.TestCase):
    """Test single profile retrieval."""

    def setUp(self):
        self.config_path = _write_test_config(_TEST_CONFIG)

    def tearDown(self):
        os.unlink(self.config_path)

    def test_get_existing(self):
        p = get_profile("test_profile_a", self.config_path)
        self.assertEqual(p.model_path, "test/model-a")

    def test_get_missing(self):
        with self.assertRaises(KeyError):
            get_profile("nonexistent", self.config_path)

    def test_get_default(self):
        p = get_default_profile(self.config_path)
        self.assertEqual(p.name, "test_profile_a")


class TestPromotionGates(unittest.TestCase):
    """Test promotion gate loading and checking."""

    def setUp(self):
        self.config_path = _write_test_config(_TEST_CONFIG)

    def tearDown(self):
        os.unlink(self.config_path)

    def test_gate_count(self):
        gates = get_promotion_gates(self.config_path)
        self.assertEqual(len(gates), 5)

    def test_min_gate(self):
        gate = PromotionGate("compile_strict_rate", "min", 95.0)
        self.assertTrue(gate.passes(95.0))
        self.assertTrue(gate.passes(100.0))
        self.assertFalse(gate.passes(94.9))

    def test_max_gate(self):
        gate = PromotionGate("fallback_rate", "max", 5.0)
        self.assertTrue(gate.passes(5.0))
        self.assertTrue(gate.passes(0.0))
        self.assertFalse(gate.passes(5.1))

    def test_check_all_pass(self):
        metrics = {
            "compile_strict_rate": 96.0,
            "compile_permissive_rate": 98.0,
            "runtime_unverified_compile_rate": 1.0,
            "fallback_rate": 3.0,
            "health_logger_scenario_score": 0.90,
        }
        result = check_promotion(metrics, self.config_path)
        self.assertTrue(result["passed"])
        self.assertEqual(result["gates_passed"], 5)

    def test_check_partial_fail(self):
        metrics = {
            "compile_strict_rate": 90.0,  # below 95
            "compile_permissive_rate": 98.0,
            "runtime_unverified_compile_rate": 1.0,
            "fallback_rate": 3.0,
            "health_logger_scenario_score": 0.90,
        }
        result = check_promotion(metrics, self.config_path)
        self.assertFalse(result["passed"])
        self.assertEqual(result["gates_passed"], 4)

    def test_check_missing_metric(self):
        metrics = {
            "compile_strict_rate": 96.0,
            # Missing others
        }
        result = check_promotion(metrics, self.config_path)
        self.assertFalse(result["passed"])

    def test_gate_string(self):
        gate = PromotionGate("test_metric", "min", 95.0)
        self.assertEqual(str(gate), "test_metric >= 95.0")
        gate2 = PromotionGate("test_metric", "max", 5.0)
        self.assertEqual(str(gate2), "test_metric <= 5.0")


class TestListProfiles(unittest.TestCase):
    """Test profile listing for MCP."""

    def setUp(self):
        self.config_path = _write_test_config(_TEST_CONFIG)

    def tearDown(self):
        os.unlink(self.config_path)

    def test_list_format(self):
        profiles = list_profiles(self.config_path)
        self.assertEqual(len(profiles), 2)
        names = {p["name"] for p in profiles}
        self.assertEqual(names, {"test_profile_a", "test_profile_b"})

    def test_default_flag(self):
        profiles = list_profiles(self.config_path)
        defaults = [p for p in profiles if p["is_default"]]
        self.assertEqual(len(defaults), 1)
        self.assertEqual(defaults[0]["name"], "test_profile_a")


class TestResolvedAdapterPath(unittest.TestCase):
    """Test adapter path resolution."""

    def test_none_adapter(self):
        p = ModelProfile(name="test", model_path="m", adapter_path=None, chat_template="llama3")
        self.assertIsNone(p.resolved_adapter_path())

    def test_absolute_adapter(self):
        p = ModelProfile(name="test", model_path="m", adapter_path="/abs/path", chat_template="llama3")
        self.assertEqual(p.resolved_adapter_path(), "/abs/path")

    def test_relative_adapter_nonexistent(self):
        p = ModelProfile(name="test", model_path="m", adapter_path="models/nonexistent", chat_template="llama3")
        # Falls back to the raw path when resolved doesn't exist
        result = p.resolved_adapter_path()
        self.assertIsNotNone(result)


class TestRealConfig(unittest.TestCase):
    """Test loading the actual project config (if it exists)."""

    def test_load_real_config(self):
        real_path = Path(__file__).resolve().parent.parent / "configs" / "model_profiles.yaml"
        if not real_path.exists():
            self.skipTest("Real config not found")

        config = load_profiles(real_path)
        self.assertGreater(len(config.profiles), 0)
        self.assertIn(config.default_profile, config.profiles)
        self.assertGreater(len(config.promotion_gates), 0)

        # Check known profiles exist
        self.assertIn("local_8b_mainline", config.profiles)

        # Validate promotion gates have reasonable thresholds
        for gate in config.promotion_gates:
            self.assertGreater(gate.threshold, 0)


if __name__ == "__main__":
    unittest.main()
