#!/usr/bin/env python3
"""Smoke end-to-end test: fixture -> vocab -> model -> forward -> eval."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_train.jsonl"


class TestSmokeE2E(unittest.TestCase):
    """Smoke test verifying the full pipeline can run on tiny data."""

    def test_fixture_loads(self):
        """Fixture JSONL loads as valid TypedIRExample records."""
        from research.src.data import load_typed_ir_jsonl
        examples = load_typed_ir_jsonl(FIXTURE)
        self.assertEqual(len(examples), 3)
        for ex in examples:
            self.assertGreater(len(ex.tier1_tokens), 0)
            self.assertGreater(len(ex.tier2_blocks), 0)

    def test_fixture_roundtrip_lower(self):
        """Each fixture example round-trips through lowering -> parse."""
        try:
            from research.src.data import load_typed_ir_jsonl
            from research.src.lowering import roundtrip_validate
            examples = load_typed_ir_jsonl(FIXTURE)
            for ex in examples:
                success, msg = roundtrip_validate(ex)
                self.assertTrue(success, f"Roundtrip failed for {ex.shortcut_id}: {msg}")
        except ImportError:
            self.skipTest("lowering not available")

    def test_vocab_from_fixture(self):
        """Can build tier1 vocab from fixture data."""
        from research.src.data import load_typed_ir_jsonl
        examples = load_typed_ir_jsonl(FIXTURE)
        # Simple vocab extraction
        all_tokens = set()
        for ex in examples:
            all_tokens.update(ex.tier1_tokens)
        self.assertIn("SHORTCUT", all_tokens)
        self.assertIn("ACTION", all_tokens)
        self.assertIn("ENDSHORTCUT", all_tokens)

    def test_model_forward_pass(self):
        """TernaryDecoder forward pass with tiny dimensions."""
        try:
            import torch
            from research.src.ternary_decoder import TernaryDecoder
            decoder = TernaryDecoder(
                input_dim=32,
                hidden_dim=16,
                tier1_vocab_size=10,
                tier2_vocab_size=20,
                num_layers=1,
            )
            x = torch.randn(2, 32)
            result = decoder(x)
            self.assertEqual(result["tier1_logits"].shape, (2, 10))
            self.assertEqual(result["tier2_logits"].shape, (2, 20))
        except ImportError:
            self.skipTest("torch not available")

    def test_losses_compute(self):
        """CompositeLoss runs without NaN on random data."""
        try:
            import torch
            from research.src.losses import CompositeLoss
            loss_fn = CompositeLoss(initial_log_sigma=0.0)
            logits = torch.randn(4, 10)
            targets = torch.randint(0, 10, (4,))
            pos = torch.randn(4, 32)
            neg = torch.randn(4, 32)
            result = loss_fn(logits, targets, pos, neg)
            self.assertFalse(torch.isnan(result["L_total"]).any())
        except ImportError:
            self.skipTest("torch not available")


if __name__ == "__main__":
    unittest.main()
