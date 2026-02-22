"""Tests for scripts/build_decoder_vocab.py — vocabulary building and coverage."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: allow importing research.src and scripts
# ---------------------------------------------------------------------------
_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RESEARCH_ROOT.parent
_SCRIPTS = _RESEARCH_ROOT / "scripts"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Mock torch before importing modules that depend on it (research.src.data)
if "torch" not in sys.modules:
    _mock_torch = types.ModuleType("torch")
    _mock_torch_utils = types.ModuleType("torch.utils")
    _mock_torch_utils_data = types.ModuleType("torch.utils.data")
    _mock_torch_utils_data.Dataset = type("Dataset", (), {})  # type: ignore[attr-defined]
    _mock_torch.utils = _mock_torch_utils  # type: ignore[attr-defined]
    _mock_torch_utils.data = _mock_torch_utils_data  # type: ignore[attr-defined]
    sys.modules["torch"] = _mock_torch
    sys.modules["torch.utils"] = _mock_torch_utils
    sys.modules["torch.utils.data"] = _mock_torch_utils_data

from build_decoder_vocab import (
    MIN_ACTION_EXAMPLES,
    SPECIAL_TOKENS,
    _build_tier1_vocab,
    _build_tier2_vocabs,
    _enforce_gates,
    _measure_tier1_coverage,
)
from research.src.contracts import CoverageReport, Tier2Block, TypedIRExample

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_example(
    tier1_tokens: list[str],
    tier2_blocks: list[Tier2Block] | None = None,
) -> TypedIRExample:
    """Create a minimal TypedIRExample for testing."""
    return TypedIRExample(
        shortcut_id="test-001",
        system_prompt="sys",
        prompt="test prompt",
        dsl="DSL",
        shortcut_name="TestShortcut",
        tier1_tokens=tier1_tokens,
        tier2_blocks=tier2_blocks or [],
        tier3_slots=[],
    )


def _make_block(action_name: str, tokens: list[str], index: int = 0) -> Tier2Block:
    """Create a minimal Tier2Block for testing."""
    return Tier2Block(action_index=index, action_name=action_name, tokens=tokens)


# ---------------------------------------------------------------------------
# Tests: _build_tier1_vocab
# ---------------------------------------------------------------------------


class TestBuildTier1Vocab:
    """Test tier1 vocabulary building."""

    def test_includes_special_tokens(self):
        """Special tokens (<PAD>, <UNK>, <BOS>, <EOS>) must appear at indices 0-3."""
        examples = [_make_example(tier1_tokens=["SHORTCUT", "ENDSHORTCUT"])]
        vocab = _build_tier1_vocab(examples)
        for token, idx in SPECIAL_TOKENS.items():
            assert token in vocab
            assert vocab[token] == idx

    def test_includes_real_tokens(self):
        """Real tokens from training examples should appear in the vocabulary."""
        examples = [
            _make_example(tier1_tokens=["SHORTCUT", "IF", "ENDSHORTCUT"]),
            _make_example(tier1_tokens=["SHORTCUT", "REPEAT", "ENDSHORTCUT"]),
        ]
        vocab = _build_tier1_vocab(examples)
        for tok in ["SHORTCUT", "IF", "ENDSHORTCUT", "REPEAT"]:
            assert tok in vocab, f"Expected token '{tok}' in vocab"

    def test_real_tokens_start_at_correct_index(self):
        """Real tokens should start at FIRST_REAL_INDEX (4)."""
        examples = [_make_example(tier1_tokens=["ALPHA", "BETA"])]
        vocab = _build_tier1_vocab(examples)
        real_indices = [v for k, v in vocab.items() if k not in SPECIAL_TOKENS]
        assert min(real_indices) == len(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Tests: _build_tier2_vocabs
# ---------------------------------------------------------------------------


class TestBuildTier2Vocabs:
    """Test tier2 per-action and fallback vocabulary building."""

    def test_creates_per_action_vocabs_for_frequent_actions(self):
        """Actions with >= MIN_ACTION_EXAMPLES blocks get their own vocab."""
        blocks = [
            _make_block("action.frequent", ["param_a", "param_b"], index=i)
            for i in range(MIN_ACTION_EXAMPLES)
        ]
        examples = [_make_example(tier1_tokens=["T"], tier2_blocks=blocks)]
        per_action, _fallback, counts = _build_tier2_vocabs(examples)

        assert "action.frequent" in per_action
        assert "param_a" in per_action["action.frequent"]
        assert "param_b" in per_action["action.frequent"]
        assert counts["action.frequent"] == MIN_ACTION_EXAMPLES

    def test_rare_actions_go_to_global_fallback_only(self):
        """Actions with < MIN_ACTION_EXAMPLES blocks should NOT get a dedicated vocab,
        but their tokens should appear in the global fallback."""
        blocks = [
            _make_block("action.rare", ["rare_tok_1", "rare_tok_2"], index=0),
        ]
        examples = [_make_example(tier1_tokens=["T"], tier2_blocks=blocks)]
        per_action, fallback, counts = _build_tier2_vocabs(examples)

        assert "action.rare" not in per_action
        assert "rare_tok_1" in fallback
        assert "rare_tok_2" in fallback
        assert counts["action.rare"] == 1

    def test_global_fallback_includes_all_tokens(self):
        """Global fallback should contain tokens from both frequent and rare actions."""
        frequent_blocks = [
            _make_block("action.freq", ["shared_tok"], index=i) for i in range(MIN_ACTION_EXAMPLES)
        ]
        rare_blocks = [_make_block("action.rare", ["only_rare"], index=0)]
        examples = [
            _make_example(tier1_tokens=["T"], tier2_blocks=frequent_blocks + rare_blocks),
        ]
        _per_action, fallback, _counts = _build_tier2_vocabs(examples)

        assert "shared_tok" in fallback
        assert "only_rare" in fallback


# ---------------------------------------------------------------------------
# Tests: _measure_tier1_coverage
# ---------------------------------------------------------------------------


class TestMeasureTier1Coverage:
    """Test tier1 coverage measurement."""

    def test_100_percent_coverage(self):
        """When all eval tokens are in the vocab, coverage should be 100%."""
        train = [_make_example(tier1_tokens=["A", "B", "C"])]
        vocab = _build_tier1_vocab(train)
        eval_ex = [_make_example(tier1_tokens=["A", "B"])]

        report = _measure_tier1_coverage(eval_ex, vocab)
        assert report.coverage_pct == 100.0
        assert report.uncovered == []
        assert report.covered == 2
        assert report.total_tokens_in_eval == 2

    def test_uncovered_tokens(self):
        """Eval tokens not in the vocab should be reported as uncovered."""
        train = [_make_example(tier1_tokens=["A", "B"])]
        vocab = _build_tier1_vocab(train)
        eval_ex = [_make_example(tier1_tokens=["A", "B", "MISSING"])]

        report = _measure_tier1_coverage(eval_ex, vocab)
        assert report.coverage_pct < 100.0
        assert "MISSING" in report.uncovered
        assert report.covered == 2
        assert report.total_tokens_in_eval == 3

    def test_empty_eval_gives_100(self):
        """Empty eval set should return 100% coverage."""
        train = [_make_example(tier1_tokens=["A"])]
        vocab = _build_tier1_vocab(train)
        eval_ex = [_make_example(tier1_tokens=[])]

        report = _measure_tier1_coverage(eval_ex, vocab)
        assert report.coverage_pct == 100.0


# ---------------------------------------------------------------------------
# Tests: _enforce_gates
# ---------------------------------------------------------------------------


class TestEnforceGates:
    """Test coverage gate enforcement."""

    def test_passes_when_above_thresholds(self):
        """Should not exit when both coverages are above their thresholds."""
        tier1_cov = CoverageReport(
            scope="tier1",
            dataset="eval",
            total_tokens_in_eval=100,
            covered=99,
            uncovered=["X"],
            coverage_pct=99.0,
        )
        tier2_cov = CoverageReport(
            scope="tier2",
            dataset="eval",
            total_tokens_in_eval=200,
            covered=196,
            uncovered=["Y1", "Y2", "Y3", "Y4"],
            coverage_pct=98.0,
        )
        # Should NOT raise SystemExit
        _enforce_gates(tier1_cov, tier2_cov)

    def test_fails_tier1_below_threshold(self):
        """Should sys.exit(1) when tier1 coverage < 98%."""
        tier1_cov = CoverageReport(
            scope="tier1",
            dataset="eval",
            total_tokens_in_eval=100,
            covered=90,
            uncovered=["X"] * 10,
            coverage_pct=90.0,
        )
        tier2_cov = CoverageReport(
            scope="tier2",
            dataset="eval",
            total_tokens_in_eval=200,
            covered=200,
            uncovered=[],
            coverage_pct=100.0,
        )
        with pytest.raises(SystemExit) as exc_info:
            _enforce_gates(tier1_cov, tier2_cov)
        assert exc_info.value.code == 1

    def test_fails_tier2_below_threshold(self):
        """Should sys.exit(1) when tier2 coverage < 95%."""
        tier1_cov = CoverageReport(
            scope="tier1",
            dataset="eval",
            total_tokens_in_eval=100,
            covered=100,
            uncovered=[],
            coverage_pct=100.0,
        )
        tier2_cov = CoverageReport(
            scope="tier2",
            dataset="eval",
            total_tokens_in_eval=100,
            covered=80,
            uncovered=["Z"] * 20,
            coverage_pct=80.0,
        )
        with pytest.raises(SystemExit) as exc_info:
            _enforce_gates(tier1_cov, tier2_cov)
        assert exc_info.value.code == 1
