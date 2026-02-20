"""Centralized path resolution for ShortcutForge.

All modules should import paths from here rather than computing them locally.
This module resolves paths relative to the project root (parent of src/).
"""

from __future__ import annotations

from pathlib import Path

# Project root: parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
SRC_DIR = PROJECT_ROOT / "src"
CLI_DIR = PROJECT_ROOT / "cli"
TESTS_DIR = PROJECT_ROOT / "tests"
TRAINING_DIR = PROJECT_ROOT / "training"
TOOLS_DIR = PROJECT_ROOT / "tools"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Data and reference directories
REFERENCES_DIR = PROJECT_ROOT / "references"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOWNLOADED_DIR = PROJECT_ROOT / "downloaded"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Key reference files
ACTION_CATALOG_PATH = REFERENCES_DIR / "action_catalog.json"
PARAM_SCHEMAS_PATH = REFERENCES_DIR / "param_schemas.json"
DSL_GRAMMAR_PATH = REFERENCES_DIR / "shortcutdsl.lark"
OUTLINES_GRAMMAR_PATH = REFERENCES_DIR / "shortcutdsl_outlines.lark"
MACRO_PATTERNS_PATH = REFERENCES_DIR / "macro_patterns.json"
SNIPPET_REGISTRY_PATH = REFERENCES_DIR / "snippet_registry.json"
DOMAIN_PROFILES_DIR = REFERENCES_DIR / "domain_profiles"
SCENARIO_PACKS_DIR = REFERENCES_DIR / "scenario_packs"

# Key data files
TRAIN_DATA_PATH = TRAINING_DATA_DIR / "shortcutdsl_train_expanded.jsonl"
EVAL_DATA_PATH = TRAINING_DATA_DIR / "shortcutdsl_eval.jsonl"
BASELINE_SNAPSHOT_PATH = TRAINING_DATA_DIR / "baseline_snapshot.json"
EVAL_RESULTS_PATH = TRAINING_DATA_DIR / "eval_results.json"
DISTILLATION_LOG_PATH = TRAINING_DATA_DIR / "distillation_log.jsonl"

# Model profiles
MODEL_PROFILES_PATH = CONFIGS_DIR / "model_profiles.yaml"
