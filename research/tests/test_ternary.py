#!/usr/bin/env python3
"""Tests for ternary quantization — values in {-1, 0, +1} and STE backward finite."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from research.src.ternary_decoder import ternary_quantize


class TestTernaryQuantize(unittest.TestCase):
    """Tests for the ternary_quantize function.

    These tests verify the contract: output must contain only {-1, 0, +1}
    and STE backward must produce finite gradients. Implementation-independent
    until Phase 0/1 fills in the function.
    """

    def test_output_values_only_ternary(self):
        """Quantized weights must be exactly {-1, 0, +1}."""
        w = torch.randn(32, 64)
        try:
            q = ternary_quantize(w)
        except NotImplementedError:
            self.skipTest("ternary_quantize not yet implemented")
        unique = set(q.unique().tolist())
        self.assertTrue(
            unique.issubset({-1.0, 0.0, 1.0}),
            f"Expected only {{-1, 0, 1}}, got {unique}",
        )

    def test_ste_backward_finite(self):
        """STE backward through quantization must produce finite gradients."""
        w = torch.randn(16, 32, requires_grad=True)
        try:
            q = ternary_quantize(w)
        except NotImplementedError:
            self.skipTest("ternary_quantize not yet implemented")
        loss = q.sum()
        loss.backward()
        self.assertFalse(torch.isnan(w.grad).any(), "STE gradient contains NaN")
        self.assertFalse(torch.isinf(w.grad).any(), "STE gradient contains Inf")

    def test_quantize_deterministic(self):
        """Same input should produce same output."""
        w = torch.randn(8, 16)
        try:
            q1 = ternary_quantize(w)
        except NotImplementedError:
            self.skipTest("ternary_quantize not yet implemented")
        q2 = ternary_quantize(w)
        self.assertTrue(torch.equal(q1, q2))


if __name__ == "__main__":
    unittest.main()
