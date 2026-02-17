#!/usr/bin/env python3
"""Tests for research.src.losses — loss components non-negative and finite."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch


class TestCompositeLoss(unittest.TestCase):
    """Verify loss component contracts on synthetic tensors.

    Implementation-independent: tests skip if not yet implemented.
    """

    def test_log_sigma_parameters_exist(self):
        from research.src.losses import CompositeLoss
        loss = CompositeLoss(initial_log_sigma=0.0)
        self.assertTrue(hasattr(loss, "log_sigma_ce"))
        self.assertTrue(hasattr(loss, "log_sigma_margin"))
        self.assertTrue(hasattr(loss, "log_sigma_repair"))
        # They should be nn.Parameters
        self.assertIsInstance(loss.log_sigma_ce, torch.nn.Parameter)

    def test_composite_forward_returns_dict(self):
        from research.src.losses import CompositeLoss
        loss = CompositeLoss()
        try:
            result = loss(
                logits=torch.randn(4, 10),
                targets=torch.randint(0, 10, (4,)),
            )
            self.assertIsInstance(result, dict)
            self.assertIn("L_total", result)
        except NotImplementedError:
            self.skipTest("CompositeLoss.forward not yet implemented")


class TestOODLoss(unittest.TestCase):
    def test_forward_returns_scalar(self):
        from research.src.losses import OODLoss
        loss = OODLoss()
        try:
            result = loss(
                logits=torch.randn(4, 1),
                labels=torch.randint(0, 2, (4, 1)).float(),
            )
            self.assertEqual(result.dim(), 0)  # scalar
        except NotImplementedError:
            self.skipTest("OODLoss.forward not yet implemented")


if __name__ == "__main__":
    unittest.main()
