"""Model-agnostic LoRA fine-tuning engine with PAB integration."""

from __future__ import annotations

import json
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.behavioral_fingerprint import BehavioralFingerprint
from src.dsl_evaluator import DSLMetrics, evaluate_outputs
from src.lora_utils import detect_target_modules, load_jsonl_data
from src.pab_tracker import CheckpointData, PABTracker

if TYPE_CHECKING:
    from src.model_registry import ModelSpec

logger = logging.getLogger(__name__)

_MAX_EVAL_SAMPLES = 100

# Zero-cost containers for config grouping
_LoRAHyperparams = namedtuple("_LoRAHyperparams", "rank alpha lr batch_size max_steps")
_CheckpointSchedule = namedtuple("_CheckpointSchedule", "checkpoint_interval eval_interval")


@dataclass
class LoRATrainConfig:
    """Configuration for LoRA fine-tuning."""

    model_id: str = ""
    train_file: Path = field(default_factory=lambda: Path("training_data/train.jsonl"))
    eval_file: Path = field(default_factory=lambda: Path("training_data/eval.jsonl"))
    hparams: _LoRAHyperparams = field(
        default_factory=lambda: _LoRAHyperparams(
            rank=16, alpha=32, lr=2e-4, batch_size=4, max_steps=1000
        )
    )
    schedule: _CheckpointSchedule = field(
        default_factory=lambda: _CheckpointSchedule(checkpoint_interval=50, eval_interval=100)
    )
    target_modules: list[str] | None = None
    output_dir: Path = field(default_factory=lambda: Path("research/results/phase_b"))
    device: str = "mps"


class LoRATrainer:
    """LoRA fine-tuning engine with PAB checkpoint recording."""

    def __init__(self, config: LoRATrainConfig, model_spec: ModelSpec) -> None:
        self.config = config
        self.model_spec = model_spec
        self.model: Any = None
        self.tokenizer: Any = None
        self.optimizer: Any = None
        self.tracker: PABTracker | None = None
        self.train_data: list[dict] = []
        self.eval_data: list[dict] = []
        self.fingerprints: list[BehavioralFingerprint] = []
        self._step = 0

    def setup(self) -> None:
        """Load model, apply LoRA, prepare data and tracker."""
        import torch
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model %s", self.model_spec.hf_repo_or_path)
        model_path = self.model_spec.local_path or self.model_spec.hf_repo_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_spec.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=self.model_spec.trust_remote_code,
        )

        target_modules = self.config.target_modules or self._detect_target_modules(self.model)
        logger.info("LoRA target modules: %s", target_modules)

        lora_config = LoraConfig(
            r=self.config.hparams.rank,
            lora_alpha=self.config.hparams.alpha,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.config.device)
        self.model.print_trainable_parameters()

        self.train_data = self._load_data(self.config.train_file)
        self.eval_data = self._load_data(self.config.eval_file)

        self.tracker = PABTracker(
            experiment_id=self.config.model_id,
            config_hash=f"lora_r{self.config.hparams.rank}_a{self.config.hparams.alpha}",
            checkpoint_interval=self.config.schedule.checkpoint_interval,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.hparams.lr,
            weight_decay=0.01,
        )

    @staticmethod
    def _detect_target_modules(model: Any) -> list[str]:
        """Delegate to lora_utils.detect_target_modules."""
        return detect_target_modules(model)

    def _tokenize_chat(self, messages: list[dict]) -> dict[str, Any]:
        """Tokenize a chat-format message list into model inputs."""
        import torch

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            if isinstance(encoded, dict):
                return {k: v.to(self.config.device) for k, v in encoded.items()}
            return {
                "input_ids": encoded.to(self.config.device),
                "attention_mask": torch.ones_like(encoded).to(self.config.device),
            }
        except Exception:
            # Fallback: simple concatenation for tokenizers without chat templates
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}\n")
            text = "".join(parts)
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            )
            return {k: v.to(self.config.device) for k, v in encoded.items()}

    @staticmethod
    def _load_data(path: Path) -> list[dict]:
        """Delegate to lora_utils.load_jsonl_data."""
        return load_jsonl_data(path)

    def train_step(self, batch: list[dict]) -> dict[str, float]:
        """Execute a single training step on a batch of examples."""
        import torch

        self.model.train()
        total_loss = 0.0

        for example in batch:
            messages = example.get("messages", [])
            if not messages:
                continue

            inputs = self._tokenize_chat(messages)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / len(batch)
            loss.backward()
            total_loss += loss.item() * len(batch)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": total_loss / max(len(batch), 1)}

    def evaluate(self, step: int) -> dict:
        """Run evaluation on the eval set and compute DSL + fingerprint metrics."""
        self.model.eval()
        eval_samples = self.eval_data[:_MAX_EVAL_SAMPLES]
        if not eval_samples:
            return {"dsl_metrics": DSLMetrics().__dict__, "step": step}

        # Extract prompts and generate
        prompts = []
        for ex in eval_samples:
            messages = ex.get("messages", [])
            user_msgs = [m for m in messages if m.get("role") == "user"]
            prompts.append(user_msgs[-1]["content"] if user_msgs else "")

        outputs = self._generate_batch(prompts)

        # Compute DSL metrics
        dsl_metrics = evaluate_outputs(outputs, eval_samples)

        # Build behavioral fingerprint from action predictions
        action_preds = []
        for out in outputs:
            from src.dsl_evaluator import _extract_actions

            actions = _extract_actions(out)
            action_preds.append(actions[0] if actions else "NONE")

        fp = BehavioralFingerprint.from_text_only(
            experiment_id=self.config.model_id,
            step=step,
            action_predictions=action_preds,
            probe_labels=prompts[: len(action_preds)],
        )
        self.fingerprints.append(fp)

        return {
            "step": step,
            "dsl_metrics": dsl_metrics.__dict__,
            "fingerprint": fp._to_dict(),
        }

    def _generate_batch(self, prompts: list[str], max_tokens: int = 256) -> list[str]:
        """Generate text outputs for a batch of prompts."""
        import torch

        self.model.eval()
        results = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.config.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                # Decode only the generated tokens
                gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

        return results

    def _record_pab_checkpoint(self, step: int, loss: float, eval_result: dict | None) -> None:
        """Record a PAB checkpoint with available metrics."""
        if self.tracker is None:
            return

        tier_accs = None
        if eval_result and "dsl_metrics" in eval_result:
            dm = eval_result["dsl_metrics"]
            tier_accs = {
                "tier1": dm.get("first_action_accuracy", 0.0),
                "tier2": dm.get("parse_rate", 0.0),
                "tier3": dm.get("endshortcut_rate", 0.0),
            }

        data = CheckpointData(
            step=step,
            train_loss=loss,
            tier_accuracies=tier_accs,
        )
        self.tracker.record(data)

    def train(self) -> dict:
        """Run main training loop with PAB profiling; return summary dict."""
        if not self.train_data:
            logger.error("No training data loaded")
            return {"error": "no training data"}

        rng = np.random.default_rng(42)
        n = len(self.train_data)
        bs = self.config.hparams.batch_size
        latest_loss = 0.0
        latest_eval: dict | None = None

        output_dir = self.config.output_dir / self.config.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting training: %d steps, batch_size=%d, data=%d examples",
            self.config.hparams.max_steps,
            bs,
            n,
        )

        for step in range(1, self.config.hparams.max_steps + 1):
            self._step = step

            # Sample a batch
            indices = rng.integers(0, n, size=bs)
            batch = [self.train_data[i] for i in indices]

            step_result = self.train_step(batch)
            latest_loss = step_result["loss"]

            if step % 10 == 0:
                logger.info(
                    "Step %d/%d  loss=%.4f", step, self.config.hparams.max_steps, latest_loss
                )

            # Evaluation
            if step % self.config.schedule.eval_interval == 0:
                latest_eval = self.evaluate(step)
                logger.info("Eval at step %d: %s", step, latest_eval.get("dsl_metrics", {}))

            # PAB checkpoint
            if step % self.config.schedule.checkpoint_interval == 0:
                self._record_pab_checkpoint(step, latest_loss, latest_eval)

                if self.tracker and self.tracker.should_early_exit(step):
                    logger.warning("PAB early exit triggered at step %d", step)
                    break

        return self._finalize_training(output_dir, latest_loss, latest_eval)

    def _finalize_training(
        self, output_dir: Path, latest_loss: float, latest_eval: dict | None
    ) -> dict:
        """Save artifacts and return training summary."""
        self.save_adapter(output_dir / "adapter")

        if self.tracker:
            profile = self.tracker.finalize()
            profile.save(output_dir / "pab_profile.json")

        for fp in self.fingerprints:
            fp.save(output_dir / f"fingerprint_step{fp.step}.json")

        summary: dict = {
            "model_id": self.config.model_id,
            "final_step": self._step,
            "final_loss": latest_loss,
            "output_dir": str(output_dir),
            "pab_profile_path": str(output_dir / "pab_profile.json"),
            "num_fingerprints": len(self.fingerprints),
        }
        if latest_eval and "dsl_metrics" in latest_eval:
            summary["final_dsl_metrics"] = latest_eval["dsl_metrics"]

        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_adapter(self, path: Path) -> None:
        """Save the PEFT adapter weights."""
        if self.model is None:
            logger.warning("No model to save")
            return
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        logger.info("Adapter saved to %s", path)
