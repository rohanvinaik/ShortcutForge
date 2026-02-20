"""
Training loop for Balanced Sashimi models.

Handles config loading, gradient safety checks, ternary distribution
logging, NaN abort, and checkpoint management.

Usage:
    uv run python research/src/trainer.py --config research/configs/base.yaml --run-id test-v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


class BalancedSashimiTrainer:
    """Training orchestrator for the Balanced Sashimi pipeline.

    Handles:
        - Config-driven model assembly
        - Gradient health checks (every N steps)
        - NaN/Inf abort with checkpoint + diagnostics
        - Ternary weight distribution logging per layer
        - Checkpoint save/resume

    Args:
        config: Experiment configuration dict.
        run_id: Unique identifier for this training run.
        device: Target device ("mps", "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: dict,
        run_id: str,
        device: str = "mps",
        seed: int = 42,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.device = device
        self.seed = seed

    def setup(self) -> None:
        """Initialize model, optimizer, data loaders, and logging."""
        import json

        import torch
        from torch.utils.data import DataLoader

        from research.src.bridge import InformationBridge
        from research.src.data import TypedIRDataset
        from research.src.domain_gate import DomainGate
        from research.src.encoder import PromptEncoder
        from research.src.intent_extractor import IntentExtractor
        from research.src.losses import CompositeLoss, OODLoss
        from research.src.ternary_decoder import TernaryDecoder

        torch.manual_seed(self.seed)

        # Load vocabs
        data_cfg = self.config["data"]
        tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
        tier2_vocab_dir = PROJECT_ROOT / data_cfg["tier2_vocab_dir"]
        # Count total tier2 vocab size (use global fallback for simplicity)
        tier2_fallback = json.loads(
            (tier2_vocab_dir / "_global_fallback.json").read_text()
        )

        model_cfg = self.config["model"]

        # Instantiate modules
        self.encoder = PromptEncoder(
            model_name=model_cfg["encoder"]["model_name"],
            device=self.device,
        )
        self.domain_gate = DomainGate(
            input_dim=model_cfg["encoder"]["output_dim"],
            hidden_dim=model_cfg["domain_gate"]["hidden_dim"],
        ).to(self.device)
        self.intent_extractor = IntentExtractor(
            input_dim=model_cfg["encoder"]["output_dim"],
            frame_dim=model_cfg["intent_extractor"]["frame_dim"],
        ).to(self.device)
        self.bridge = InformationBridge(
            input_dim=model_cfg["intent_extractor"]["frame_dim"],
            bridge_dim=model_cfg["bridge"]["bridge_dim"],
        ).to(self.device)
        self.decoder = TernaryDecoder(
            input_dim=model_cfg["bridge"]["bridge_dim"],
            hidden_dim=model_cfg["decoder"]["hidden_dim"],
            tier1_vocab_size=len(tier1_vocab),
            tier2_vocab_size=len(tier2_fallback),
            num_layers=model_cfg["decoder"]["num_layers"],
        ).to(self.device)

        # Loss functions
        train_cfg = self.config["training"]
        self.composite_loss = CompositeLoss(
            initial_log_sigma=train_cfg["loss"]["initial_log_sigma"],
        ).to(self.device)
        self.ood_loss = OODLoss().to(self.device)

        # Optimizer — only non-frozen params
        params = list(self._trainable_params())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Data loaders
        self.train_dataset = TypedIRDataset(PROJECT_ROOT / data_cfg["typed_ir_train"])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

        # Vocab references for target generation
        self.tier1_vocab = tier1_vocab
        self.tier2_vocab = tier2_fallback
        self.tier1_token2idx = tier1_vocab  # already token->idx
        self.tier2_token2idx = tier2_fallback

        # Logging setup
        self.step = 0
        self.run_dir = PROJECT_ROOT / self.config["logging"]["run_dir"] / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = PROJECT_ROOT / self.config["logging"]["checkpoint_dir"]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _trainable_params(self) -> list:
        """Collect all trainable parameters from non-frozen modules."""
        params = []
        for module in [
            self.domain_gate,
            self.intent_extractor,
            self.bridge,
            self.decoder,
            self.composite_loss,
        ]:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _build_tier1_targets(self, batch: list) -> torch.Tensor:
        """Build tier1 target indices from batch examples.

        Use the FIRST meaningful tier1 token (after SHORTCUT) as the target.
        """
        import torch

        unk_idx = self.tier1_token2idx.get("<UNK>", 1)
        targets = []
        for ex in batch:
            # Use the second token (first after SHORTCUT) or first ACTION
            if len(ex.tier1_tokens) > 1:
                token = ex.tier1_tokens[1]
            else:
                token = ex.tier1_tokens[0]
            targets.append(self.tier1_token2idx.get(token, unk_idx))
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    def train_step(self, batch: list) -> dict[str, float]:
        """Execute one training step with gradient safety checks."""
        import torch

        # Extract prompts and encode (clone to exit inference-mode context)
        prompts = [ex.prompt for ex in batch]
        embeddings = self.encoder.encode(prompts).clone().detach().requires_grad_(False)
        # Embeddings are frozen inputs — no grad needed on them, but they must
        # be normal tensors (not inference-mode) so downstream autograd works.

        # Forward through pipeline
        gate_logits = self.domain_gate(embeddings)  # [B, 1]
        frame_features = self.intent_extractor(embeddings)  # [B, 256]
        bridge_features = self.bridge(frame_features)  # [B, 128]
        decoder_output = self.decoder(
            bridge_features
        )  # dict with tier1_logits, tier2_logits

        # Build targets from examples
        tier1_targets = self._build_tier1_targets(batch)  # [B] long tensor

        # Composite loss (use bridge_features as both pos/neg for initial training)
        loss_dict = self.composite_loss(
            decoder_output["tier1_logits"],
            tier1_targets,
            bridge_features,
            bridge_features.roll(1, 0),  # shifted as neg
        )

        # Gate loss (all training examples are in-domain)
        gate_labels = torch.ones(len(batch), 1, device=self.device)
        gate_loss = self.ood_loss(gate_logits, gate_labels)

        total_loss = loss_dict["L_total"] + 0.1 * gate_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        max_grad = self.config["safety"]["max_grad_norm"]
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._trainable_params() if p.grad is not None],
            max_grad,
        )

        self.optimizer.step()
        self.step += 1

        return {
            "L_total": total_loss.item(),
            "L_ce": loss_dict["L_ce"].item(),
            "L_gate": gate_loss.item(),
            **{
                k: v.item()
                for k, v in loss_dict.items()
                if k != "L_total" and k != "L_ce" and hasattr(v, "item")
            },
        }

    def check_gradient_health(self) -> bool:
        """Check all gradients for NaN/Inf. Returns True if healthy."""
        import torch

        for name, module in [
            ("gate", self.domain_gate),
            ("intent", self.intent_extractor),
            ("bridge", self.bridge),
            ("decoder", self.decoder),
        ]:
            for pname, p in module.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"  WARNING: NaN/Inf gradient in {name}.{pname}")
                        return False
        return True

    def log_ternary_distribution(self, step: int) -> dict[str, dict[str, float]]:
        """Log {-1, 0, +1} distribution per TernaryLinear layer."""
        from research.src.ternary_decoder import TernaryLinear

        dist = {}
        for name, module in self.decoder.named_modules():
            if isinstance(module, TernaryLinear):
                w = module.weight.data
                total = w.numel()
                neg = (w < -0.01).sum().item() / total * 100
                zero = ((w >= -0.01) & (w <= 0.01)).sum().item() / total * 100
                pos = (w > 0.01).sum().item() / total * 100
                dist[name] = {"neg_pct": neg, "zero_pct": zero, "pos_pct": pos}
        return dist

    def save_checkpoint(self, step: int) -> Path:
        """Save model checkpoint with optimizer state."""
        import torch

        path = self.checkpoint_dir / f"{self.run_id}_step{step}.pt"
        torch.save(
            {
                "step": step,
                "run_id": self.run_id,
                "config": self.config,
                "domain_gate": self.domain_gate.state_dict(),
                "intent_extractor": self.intent_extractor.state_dict(),
                "bridge": self.bridge.state_dict(),
                "decoder": self.decoder.state_dict(),
                "composite_loss": self.composite_loss.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def _run_step_checks(self, loss_dict: dict, safety: dict) -> dict | None:
        """Run per-step safety checks and logging. Returns early-exit dict or None."""
        if self.step % safety["gradient_check_every_n_steps"] == 0:
            if not self.check_gradient_health() and safety["nan_abort"]:
                if safety.get("nan_checkpoint_before_abort", True):
                    self.save_checkpoint(self.step)
                print(f"ABORT: NaN/Inf gradient at step {self.step}")
                return {"status": "nan_abort", "step": self.step}

        if self.step % safety["ternary_distribution_log_every_n_steps"] == 0:
            ternary_dist = self.log_ternary_distribution(self.step)
            if ternary_dist:
                print(f"  Step {self.step} ternary: {ternary_dist}")

        if self.step % 50 == 0:
            print(f"  Step {self.step}/{self.config['training']['max_iterations']}: L_total={loss_dict['L_total']:.4f}")

        return None

    def _finalize_training(self, all_losses, epoch, elapsed, log_path) -> dict:
        """Save checkpoint, write log, return summary."""
        import json

        ckpt_path = self.save_checkpoint(self.step)
        with open(log_path, "w") as f:
            for entry in all_losses:
                f.write(json.dumps(entry) + "\n")

        print(f"\nTraining complete: {self.step} steps in {elapsed:.1f}s")
        if all_losses:
            print(f"Final loss: {all_losses[-1]['L_total']:.4f}")
        else:
            print("No losses recorded")
        print(f"Checkpoint: {ckpt_path}")

        return {
            "status": "complete",
            "steps": self.step,
            "epochs": epoch,
            "elapsed_s": round(elapsed, 1),
            "final_loss": all_losses[-1] if all_losses else {},
            "checkpoint": str(ckpt_path),
        }

    def train(self, dry_run: bool = False) -> dict:
        """Main training loop."""
        import time

        from torch.utils.data import DataLoader

        self.setup()

        max_iters = self.config["training"]["max_iterations"]
        safety = self.config["safety"]
        log_path = (
            self.run_dir
            / f"{self.run_id}{self.config['logging']['training_log_suffix']}"
        )

        print(f"Training run: {self.run_id}")
        print(f"  Device: {self.device}")
        print(f"  Max iterations: {max_iters}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Train examples: {len(self.train_dataset)}")
        if dry_run:
            print("  [DRY RUN] Will run 1 batch only.")

        start = time.time()
        all_losses: list[dict[str, float]] = []

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

        epoch = 0
        while self.step < max_iters:
            epoch += 1
            for batch in loader:
                if self.step >= max_iters:
                    break

                loss_dict = self.train_step(batch)
                all_losses.append(loss_dict)

                abort = self._run_step_checks(loss_dict, safety)
                if abort is not None:
                    return abort

                if dry_run:
                    print(f"  Dry run complete after 1 step. Loss: {loss_dict['L_total']:.4f}")
                    return {"status": "dry_run", "step": 1, "loss": loss_dict}

        return self._finalize_training(all_losses, epoch, time.time() - start, log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Balanced Sashimi training loop",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="Unique identifier for this run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Target device (default: mps)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one batch and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = BalancedSashimiTrainer(
        config=config,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
    )
    trainer.train(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
