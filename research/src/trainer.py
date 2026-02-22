"""Training loop for Balanced Sashimi models."""

from __future__ import annotations

import argparse
import hashlib
import json as _json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.pab_tracker import PABTracker
    from src.trainer_checks import (
        check_gradient_abort,
        check_gradient_health,
        collect_decoder_weight_signs,
        log_ternary_distribution,
        record_pab_checkpoint,
        save_checkpoint,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from research.src.pab_tracker import PABTracker
    from research.src.trainer_checks import (
        check_gradient_abort,
        check_gradient_health,
        collect_decoder_weight_signs,
        log_ternary_distribution,
        record_pab_checkpoint,
        save_checkpoint,
    )

_Pipeline = namedtuple("_Pipeline", "encoder domain_gate intent_extractor bridge decoder")
_Vocabs = namedtuple("_Vocabs", "tier1 tier2")
_TrainInfra = namedtuple(
    "_TrainInfra",
    (
        "composite_loss "
        "ood_loss "
        "optimizer "
        "dataset "
        "pab_tracker "
        "gate_optimizer "
        "negative_bank_by_prompt "
        "repair_weight_by_prompt "
        "ood_examples "
        "eval_examples"
    ),
)
_TrainInfra.__new__.__defaults__ = (None, None, None, None, None)
_TrainerInit = namedtuple("_TrainerInit", "config run_id device seed encoder_override")
_RunPaths = namedtuple("_RunPaths", "run_dir checkpoint_dir")

_DEFAULT_DOMAIN = "general"
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "health_logger": ("health", "workout", "steps", "caffeine", "sleep", "calories"),
    "api_pagination_fetcher": ("api", "http", "json", "request", "fetch", "endpoint"),
    "calendar_triage": ("calendar", "event", "meeting", "schedule"),
    "clipboard_utility": ("clipboard", "copy", "paste"),
    "file_router": ("file", "folder", "document", "pdf"),
    "media_metadata_pipeline": ("media", "music", "photo", "video", "metadata"),
    "morning_routine": ("morning", "alarm", "wake", "weather", "routine"),
    "share_sheet_text_cleaner": ("share", "sheet", "clean", "sanitize", "rewrite"),
}
_ACTION_DOMAIN_HINTS: dict[str, str] = {
    "health": "health_logger",
    "calendar": "calendar_triage",
    "clipboard": "clipboard_utility",
    "file": "file_router",
    "photo": "media_metadata_pipeline",
    "music": "media_metadata_pipeline",
    "video": "media_metadata_pipeline",
    "api": "api_pagination_fetcher",
    "url": "api_pagination_fetcher",
    "share": "share_sheet_text_cleaner",
}
_REPAIR_SEVERITY: dict[str, float] = {
    "validate_unknown_action": 1.4,
    "unknown_action": 1.4,
    "parse_overflow": 1.2,
    "structure": 1.1,
    "macro_expansion": 1.0,
    "action": 1.0,
    "condition": 0.9,
    "handle": 0.8,
    "interpolation": 0.7,
    "alias_warning": 0.6,
    "trailing_newline": 0.4,
}


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


class BalancedSashimiTrainer:
    """Training orchestrator for the Balanced Sashimi pipeline."""

    def __init__(
        self,
        config: dict,
        run_id: str,
        device: str = "mps",
        seed: int = 42,
        encoder_override: object | None = None,
    ) -> None:
        self._init = _TrainerInit(config, run_id, device, seed, encoder_override)
        self._paths = _RunPaths(run_dir=Path("."), checkpoint_dir=Path("."))
        self._rng = np.random.default_rng(seed)
        self._output_mode = "typed_ir"
        self._last_bridge_features: np.ndarray | None = None
        self._last_tier_accuracies: dict[str, float] = {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}
        self._last_domain_accuracies: dict[str, float] = {}
        self._last_action_accuracies: dict[str, float] = {}
        self._last_validation_snapshot: dict[str, float] = {}
        self._ood_in_domain_examples: list[Any] = []
        self._ood_out_domain_examples: list[Any] = []

    @property
    def config(self) -> dict:
        return self._init.config

    @property
    def run_id(self) -> str:
        return self._init.run_id

    @property
    def device(self) -> str:
        return self._init.device

    @property
    def seed(self) -> int:
        return self._init.seed

    @property
    def encoder_override(self) -> object | None:
        return self._init.encoder_override

    @property
    def run_dir(self) -> Path:
        return self._paths.run_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self._paths.checkpoint_dir

    def setup(self) -> None:
        """Initialize model, optimizer, data loaders, and logging."""
        import json

        import torch

        try:
            from src.bridge import InformationBridge
            from src.data import NegativeBankDataset, TypedIRDataset, load_ood_prompts_jsonl
            from src.domain_gate import DomainGate
            from src.encoder import PromptEncoder
            from src.intent_extractor import IntentExtractor
            from src.losses import CompositeLoss, OODLoss
            from src.ternary_decoder import TernaryDecoder
        except ImportError:  # pragma: no cover - fallback for direct module execution
            from research.src.bridge import InformationBridge
            from research.src.data import NegativeBankDataset, TypedIRDataset, load_ood_prompts_jsonl
            from research.src.domain_gate import DomainGate
            from research.src.encoder import PromptEncoder
            from research.src.intent_extractor import IntentExtractor
            from research.src.losses import CompositeLoss, OODLoss
            from research.src.ternary_decoder import TernaryDecoder

        torch.manual_seed(self.seed)
        data_cfg = self.config["data"]
        tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
        tier2_vocab_dir = PROJECT_ROOT / data_cfg["tier2_vocab_dir"]
        tier2_fallback = json.loads((tier2_vocab_dir / "_global_fallback.json").read_text())

        model_cfg = self.config["model"]
        decoder_cfg = model_cfg["decoder"]
        self._output_mode = decoder_cfg.get("output_mode", "typed_ir")

        encoder = (
            self.encoder_override
            if self.encoder_override is not None
            else PromptEncoder(
                model_name=model_cfg["encoder"]["model_name"],
                device=self.device,
            )
        )
        self.pipeline = _Pipeline(
            encoder=encoder,
            domain_gate=DomainGate(
                input_dim=model_cfg["encoder"]["output_dim"],
                hidden_dim=model_cfg["domain_gate"]["hidden_dim"],
            ).to(self.device),
            intent_extractor=IntentExtractor(
                input_dim=model_cfg["encoder"]["output_dim"],
                frame_dim=model_cfg["intent_extractor"]["frame_dim"],
            ).to(self.device),
            bridge=InformationBridge(
                input_dim=model_cfg["intent_extractor"]["frame_dim"],
                bridge_dim=model_cfg["bridge"]["bridge_dim"],
            ).to(self.device),
            decoder=TernaryDecoder(
                input_dim=model_cfg["bridge"]["bridge_dim"],
                hidden_dim=decoder_cfg["hidden_dim"],
                tier1_vocab_size=len(tier1_vocab),
                tier2_vocab_size=len(tier2_fallback),
                num_layers=decoder_cfg["num_layers"],
                ternary_enabled=decoder_cfg.get("ternary_enabled", True),
                partial_ternary=decoder_cfg.get("partial_ternary", False),
            ).to(self.device),
        )
        self.vocabs = _Vocabs(tier1=tier1_vocab, tier2=tier2_fallback)

        train_cfg = self.config["training"]
        loss_cfg = train_cfg.get("loss", {})
        composite_loss = CompositeLoss(
            initial_log_sigma=loss_cfg.get("initial_log_sigma", 0.0),
            margin=loss_cfg.get("margin", 0.5),
        ).to(self.device)

        pab_cfg = self.config.get("pab", {})
        pab_tracker = None
        if pab_cfg.get("enabled", False):
            config_hash = hashlib.md5(
                _json.dumps(self.config, sort_keys=True).encode()
            ).hexdigest()[:12]
            pab_tracker = PABTracker(
                experiment_id=self.run_id,
                config_hash=config_hash,
                checkpoint_interval=pab_cfg.get("checkpoint_interval", 50),
            )

        train_dataset = TypedIRDataset(PROJECT_ROOT / data_cfg["typed_ir_train"])
        negative_bank_by_prompt: dict[str, list[int]] = {}
        repair_weight_by_prompt: dict[str, float] = {}
        negative_bank_path = PROJECT_ROOT / data_cfg.get("negative_bank", "")
        if negative_bank_path.exists():
            neg_dataset = NegativeBankDataset(negative_bank_path)
            negative_bank_by_prompt, repair_weight_by_prompt = self._index_negative_bank(neg_dataset)

        ood_examples = []
        ood_path = PROJECT_ROOT / data_cfg.get("ood_prompts", "")
        if ood_path.exists():
            ood_examples = load_ood_prompts_jsonl(ood_path)
        self._ood_in_domain_examples = [p for p in ood_examples if p.label == "in_domain"]
        self._ood_out_domain_examples = [p for p in ood_examples if p.label != "in_domain"]

        eval_examples = []
        eval_path = PROJECT_ROOT / data_cfg.get("typed_ir_eval", "")
        if eval_path.exists():
            eval_examples = TypedIRDataset(eval_path).examples

        optimizer = torch.optim.AdamW(
            list(self._trainable_params(composite_loss)),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        gate_lr = train_cfg.get("domain_gate_learning_rate", train_cfg["learning_rate"])
        gate_optimizer = torch.optim.AdamW(
            list(self._gate_trainable_params()),
            lr=gate_lr,
            weight_decay=train_cfg["weight_decay"],
        )

        self.infra = _TrainInfra(
            composite_loss=composite_loss,
            ood_loss=OODLoss().to(self.device),
            optimizer=optimizer,
            dataset=train_dataset,
            pab_tracker=pab_tracker,
            gate_optimizer=gate_optimizer,
            negative_bank_by_prompt=negative_bank_by_prompt,
            repair_weight_by_prompt=repair_weight_by_prompt,
            ood_examples=ood_examples,
            eval_examples=eval_examples,
        )

        self.step = 0
        run_dir = PROJECT_ROOT / self.config["logging"]["run_dir"] / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = PROJECT_ROOT / self.config["logging"]["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._paths = _RunPaths(run_dir=run_dir, checkpoint_dir=checkpoint_dir)

    def _trainable_params(self, composite_loss=None) -> list:
        """Collect generation-side trainable parameters (excluding domain gate)."""
        loss = composite_loss if composite_loss is not None else self.infra.composite_loss
        params = []
        for module in [
            self.pipeline.intent_extractor,
            self.pipeline.bridge,
            self.pipeline.decoder,
            loss,
        ]:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _gate_trainable_params(self) -> list:
        """Collect domain gate trainable parameters."""
        return [p for p in self.pipeline.domain_gate.parameters() if p.requires_grad]

    def _build_tier1_targets(self, batch: list) -> torch.Tensor:
        """Build tier1 target indices from batch examples."""
        import torch

        unk_idx = self.vocabs.tier1.get("<UNK>", 1)
        targets = []
        for ex in batch:
            token = self._tier1_token(ex)
            targets.append(self.vocabs.tier1.get(token, unk_idx))
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    @staticmethod
    def _tier1_token(example: Any) -> str:
        """Extract first structural/action token target for an example."""
        tokens = getattr(example, "tier1_tokens", [])
        if not tokens:
            return "<UNK>"
        return tokens[1] if len(tokens) > 1 else tokens[0]

    def _index_negative_bank(self, neg_dataset: Any) -> tuple[dict[str, list[int]], dict[str, float]]:
        """Build prompt-indexed negative targets and repair weights."""
        prompt_to_negatives: dict[str, list[int]] = defaultdict(list)
        prompt_to_repair_weight: dict[str, float] = {}
        unk_idx = self.vocabs.tier1.get("<UNK>", 1)

        for entry in getattr(neg_dataset, "entries", []):
            prompt = (entry.prompt or "").strip()
            if not prompt:
                continue

            if entry.negative is not None:
                neg_token = self._tier1_token(entry.negative)
                prompt_to_negatives[prompt].append(self.vocabs.tier1.get(neg_token, unk_idx))

            sev = max((_REPAIR_SEVERITY.get(tag, 0.5) for tag in entry.error_tags), default=0.5)
            lint_count = len(entry.lint_changes)
            repair_weight = 1.0 + 0.5 * sev + 0.1 * lint_count
            prompt_to_repair_weight[prompt] = max(
                repair_weight, prompt_to_repair_weight.get(prompt, 1.0)
            )

        return dict(prompt_to_negatives), prompt_to_repair_weight

    def _sample_negative_index(self, positive_index: int) -> int:
        """Sample a random tier1 index that is different from positive_index."""
        vocab_size = max(2, len(self.vocabs.tier1))
        candidate = positive_index
        for _ in range(4):
            candidate = int(self._rng.integers(0, vocab_size))
            if candidate != positive_index:
                return candidate
        return (positive_index + 1) % vocab_size

    def _build_negative_targets(self, batch: list, tier1_targets: torch.Tensor) -> torch.Tensor:
        """Select negative targets from negative-bank map (fallback to random negatives)."""
        import torch

        negative_targets: list[int] = []
        neg_map = self.infra.negative_bank_by_prompt or {}
        for i, ex in enumerate(batch):
            positive_idx = int(tier1_targets[i].item())
            prompt = (ex.prompt or "").strip()
            candidates = neg_map.get(prompt, [])

            if candidates:
                choice = int(candidates[int(self._rng.integers(0, len(candidates)))])
                if choice == positive_idx and len(candidates) > 1:
                    choice = int(candidates[(candidates.index(choice) + 1) % len(candidates)])
                negative_targets.append(choice)
            else:
                negative_targets.append(self._sample_negative_index(positive_idx))

        return torch.tensor(negative_targets, dtype=torch.long, device=self.device)

    def _build_repair_weights(self, batch: list) -> torch.Tensor:
        """Build repair-weight vector from negative-bank tags and fallback metadata."""
        import torch

        repair_map = self.infra.repair_weight_by_prompt or {}
        weights = []
        for ex in batch:
            prompt = (ex.prompt or "").strip()
            weight = repair_map.get(prompt)
            if weight is None:
                md = getattr(ex, "metadata", {}) or {}
                weight = 1.2 if md.get("lint_applied", False) else 1.0
                tier1_len = float(md.get("tier1_len", 0))
                weight += min(0.3, tier1_len / 300.0)
            weights.append(float(weight))

        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _infer_domain(self, example: Any) -> str:
        """Infer a coarse domain label for domain-wise progression tracking."""
        prompt = (getattr(example, "prompt", "") or "").lower()
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if any(k in prompt for k in keywords):
                return domain

        blocks = getattr(example, "tier2_blocks", [])
        if blocks:
            name = (blocks[0].action_name or "").lower()
            for key, domain in _ACTION_DOMAIN_HINTS.items():
                if key in name:
                    return domain

        return _DEFAULT_DOMAIN

    def _track_batch_metrics(
        self,
        batch: list,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Track tier/domain/action metrics from the latest batch."""
        correct = predictions == targets
        tier1_acc = float(np.mean(correct)) if len(correct) else 0.0

        domain_correct: dict[str, list[float]] = defaultdict(list)
        action_correct: dict[str, list[float]] = defaultdict(list)
        idx2token = {v: k for k, v in self.vocabs.tier1.items()}
        for i, ex in enumerate(batch):
            domain = self._infer_domain(ex)
            domain_correct[domain].append(float(correct[i]))
            action_token = idx2token.get(int(targets[i]), "<UNK>")
            action_correct[action_token].append(float(correct[i]))

        self._last_tier_accuracies = {
            "tier1": tier1_acc,
            # Tier 2/3 are filled by validation snapshot when available.
            "tier2": self._last_tier_accuracies.get("tier2", tier1_acc),
            "tier3": self._last_tier_accuracies.get("tier3", tier1_acc),
        }
        self._last_domain_accuracies = {
            d: float(np.mean(vals)) for d, vals in domain_correct.items() if vals
        }
        self._last_action_accuracies = {
            a: float(np.mean(vals)) for a, vals in action_correct.items() if vals
        }

    def _sample_ood_examples(self, n: int, in_domain: bool) -> list[Any]:
        """Sample OOD prompt records by label with replacement."""
        pool = self._ood_in_domain_examples if in_domain else self._ood_out_domain_examples
        if not pool:
            return []
        indices = self._rng.integers(0, len(pool), size=n)
        return [pool[int(i)] for i in indices]

    def _domain_gate_step(self, in_domain_prompts: list[str]) -> tuple[float, float]:
        """Run a separate OOD training step for the domain gate head."""
        import torch

        gate_optimizer = self.infra.gate_optimizer
        if gate_optimizer is None:
            return 0.0, 0.0

        bs = max(1, len(in_domain_prompts))
        ood_n = max(1, bs // 2)
        id_n = max(1, bs - ood_n)

        sampled_ood = self._sample_ood_examples(ood_n, in_domain=False)
        sampled_id = self._sample_ood_examples(id_n, in_domain=True)

        prompts = list(in_domain_prompts)
        labels = [1.0] * len(in_domain_prompts)

        for rec in sampled_ood + sampled_id:
            prompts.append(rec.prompt)
            labels.append(1.0 if rec.label == "in_domain" else 0.0)

        if not prompts:
            return 0.0, 0.0

        embeddings = self.pipeline.encoder.encode(prompts).clone().detach().requires_grad_(False)
        logits = self.pipeline.domain_gate(embeddings)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

        gate_loss = self.infra.ood_loss(logits, label_tensor)
        gate_optimizer.zero_grad()
        gate_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._gate_trainable_params(), self.config["safety"]["max_grad_norm"])
        gate_optimizer.step()

        gate_pred = (torch.sigmoid(logits) > 0.5).float()
        gate_acc = float((gate_pred == label_tensor).float().mean().item())
        return float(gate_loss.item()), gate_acc

    def train_step(self, batch: list, return_per_example: bool = False) -> dict[str, float]:
        """Execute one training step with separate generation and domain-gate updates."""
        import torch
        import torch.nn.functional as F

        prompts = [ex.prompt for ex in batch]
        embeddings = self.pipeline.encoder.encode(prompts).clone().detach().requires_grad_(False)

        frame_features = self.pipeline.intent_extractor(embeddings)
        bridge_features = self.pipeline.bridge(frame_features)
        decoder_output = self.pipeline.decoder(bridge_features)
        tier1_logits = decoder_output["tier1_logits"]

        self._last_bridge_features = bridge_features.detach().cpu().numpy()
        tier1_targets = self._build_tier1_targets(batch)
        negative_targets = self._build_negative_targets(batch, tier1_targets)
        repair_weights = self._build_repair_weights(batch)

        loss_dict = self.infra.composite_loss(
            tier1_logits,
            tier1_targets,
            negative_targets=negative_targets,
            repair_weights=repair_weights,
            margin=self.config.get("training", {}).get("loss", {}).get("margin", 0.5),
        )

        total_loss = loss_dict["L_total"]
        self.infra.optimizer.zero_grad()
        total_loss.backward()

        max_grad = self.config["safety"]["max_grad_norm"]
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._trainable_params() if p.grad is not None],
            max_grad,
        )
        self.infra.optimizer.step()

        gate_loss, gate_acc = self._domain_gate_step(prompts)
        self.step += 1

        per_example_ce = F.cross_entropy(tier1_logits, tier1_targets, reduction="none")
        pred_np = tier1_logits.argmax(dim=-1).detach().cpu().numpy()
        target_np = tier1_targets.detach().cpu().numpy()
        self._track_batch_metrics(batch, pred_np, target_np)

        metrics: dict[str, float] = {
            "L_total": float(total_loss.item()),
            "L_ce": float(loss_dict["L_ce"].item()),
            "L_margin": float(loss_dict["L_margin"].item()),
            "L_repair": float(loss_dict["L_repair"].item()),
            "L_gate": float(gate_loss),
            "gate_accuracy": float(gate_acc),
            "batch_tier1_accuracy": float(self._last_tier_accuracies.get("tier1", 0.0)),
            "L_total_joint": float(total_loss.item() + 0.1 * gate_loss),
            "w_ce": float(loss_dict["w_ce"].item()),
            "w_margin": float(loss_dict["w_margin"].item()),
            "w_repair": float(loss_dict["w_repair"].item()),
            "sigma_ce": float(loss_dict["sigma_ce"].item()),
            "sigma_margin": float(loss_dict["sigma_margin"].item()),
            "sigma_repair": float(loss_dict["sigma_repair"].item()),
        }

        if return_per_example:
            metrics["per_example_loss"] = per_example_ce.detach().cpu().tolist()  # type: ignore[assignment]

        return metrics

    def _render_predicted_dsl(self, predicted_action: str, shortcut_name: str = "Generated") -> str:
        """Render a minimal deterministic DSL output for validator/compile checks."""
        name = (shortcut_name or "Generated").replace('"', '\\"')
        _ = self._output_mode  # Output mode currently shares deterministic rendering.
        return f'SHORTCUT "{name}"\nACTION {predicted_action}\nENDSHORTCUT\n'

    def _compute_validation_snapshot(self) -> dict[str, float]:
        """Compute lightweight eval metrics for PAB checkpoints."""
        import torch
        import torch.nn.functional as F

        try:
            from dsl_bridge import compile_ir
            from dsl_parser import parse_dsl
            from dsl_validator import validate_ir
        except ImportError:  # pragma: no cover - fallback for direct module execution
            return {}

        eval_examples = self.infra.eval_examples or []
        if not eval_examples:
            return {}

        pab_cfg = self.config.get("pab", {})
        subset_n = max(1, int(pab_cfg.get("validation_subset_size", 16)))
        if len(eval_examples) <= subset_n:
            subset = list(eval_examples)
        else:
            idx = self._rng.integers(0, len(eval_examples), size=subset_n)
            subset = [eval_examples[int(i)] for i in idx]

        prompts = [ex.prompt for ex in subset]
        domains = [self._infer_domain(ex) for ex in subset]

        with torch.no_grad():
            emb = self.pipeline.encoder.encode(prompts)
            frame = self.pipeline.intent_extractor(emb)
            bridge = self.pipeline.bridge(frame)
            out = self.pipeline.decoder(bridge)
            logits = out["tier1_logits"]
            targets = self._build_tier1_targets(subset)
            val_loss = float(F.cross_entropy(logits, targets).item())
            pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
            target_idx = targets.detach().cpu().numpy()

        idx2token = {v: k for k, v in self.vocabs.tier1.items()}
        predicted_tokens = [idx2token.get(int(i), "<UNK>") for i in pred_idx]

        parse_ok = 0
        validate_strict_ok = 0
        validate_perm_ok = 0
        compile_strict_ok = 0
        compile_perm_ok = 0
        runtime_unverified = 0
        domain_compile: dict[str, list[float]] = defaultdict(list)
        action_acc: dict[str, list[float]] = defaultdict(list)

        for i, ex in enumerate(subset):
            expected_action = self._tier1_token(ex)
            predicted_action = predicted_tokens[i]
            action_acc[expected_action].append(float(predicted_action == expected_action))

            dsl = self._render_predicted_dsl(predicted_action, ex.shortcut_name)
            try:
                ir = parse_dsl(dsl)
                parse_ok += 1
            except Exception:
                domain_compile[domains[i]].append(0.0)
                continue

            try:
                strict_validation = validate_ir(ir, strict=True)
                strict_valid = not strict_validation.errors
            except Exception:
                strict_valid = False

            try:
                permissive_validation = validate_ir(ir, strict=False)
                permissive_valid = not permissive_validation.errors
                has_compiler_risk = any(
                    w.category == "compiler_risk" for w in permissive_validation.warnings
                )
            except Exception:
                permissive_valid = False
                has_compiler_risk = False

            if strict_valid:
                validate_strict_ok += 1
            if permissive_valid:
                validate_perm_ok += 1

            strict_compiled = False
            perm_compiled = False
            if strict_valid:
                try:
                    compile_ir(ir)
                    strict_compiled = True
                    compile_strict_ok += 1
                except Exception:
                    strict_compiled = False

            if permissive_valid:
                if strict_compiled:
                    perm_compiled = True
                    compile_perm_ok += 1
                else:
                    try:
                        compile_ir(ir)
                        perm_compiled = True
                        compile_perm_ok += 1
                    except Exception:
                        perm_compiled = False

            if perm_compiled and (not strict_compiled or has_compiler_risk):
                runtime_unverified += 1

            domain_compile[domains[i]].append(1.0 if strict_compiled else 0.0)

        n = max(1, len(subset))
        tier1_acc = float(np.mean(pred_idx == target_idx)) if len(target_idx) else 0.0
        domain_acc = {d: float(np.mean(v)) for d, v in domain_compile.items() if v}
        action_acc_mean = {a: float(np.mean(v)) for a, v in action_acc.items() if v}

        return {
            "val_loss": val_loss,
            "tier1_accuracy": tier1_acc,
            "parse_rate": parse_ok / n,
            "validate_strict_rate": validate_strict_ok / n,
            "validate_permissive_rate": validate_perm_ok / n,
            "compile_strict_rate": compile_strict_ok / n,
            "compile_permissive_rate": compile_perm_ok / n,
            "runtime_unverified_rate": runtime_unverified / n,
            "domain_accuracies": domain_acc,  # type: ignore[dict-item]
            "action_accuracies": action_acc_mean,  # type: ignore[dict-item]
        }

    def check_gradient_health(self) -> bool:
        """Check all gradients for NaN/Inf. Returns True if healthy."""
        return check_gradient_health(self.pipeline)

    def _collect_decoder_weight_signs(self) -> np.ndarray | None:
        """Extract sign pattern from TernaryLinear layers."""
        return collect_decoder_weight_signs(self.pipeline.decoder)

    def log_ternary_distribution(self, step: int) -> dict[str, dict[str, float]]:
        """Log {-1, 0, +1} distribution per TernaryLinear layer."""
        return log_ternary_distribution(self.pipeline.decoder)

    def save_checkpoint(self, step: int) -> Path:
        """Save model checkpoint with optimizer state."""
        return save_checkpoint(
            self.checkpoint_dir,
            self.run_id,
            step,
            self.config,
            self.pipeline,
            self.infra,
        )

    def _run_step_checks(self, loss_dict: dict, safety: dict) -> dict | None:
        """Run per-step safety checks and logging. Returns early-exit dict or None."""
        abort = check_gradient_abort(self.step, safety, self.pipeline, self.save_checkpoint)
        if abort is not None:
            return abort

        if self.step % safety["ternary_distribution_log_every_n_steps"] == 0:
            ternary_dist = self.log_ternary_distribution(self.step)
            if ternary_dist:
                print(f"  Step {self.step} ternary: {ternary_dist}")

        if self.step % 50 == 0:
            print(
                f"  Step {self.step}/{self.config['training']['max_iterations']}"
                f": L_total={loss_dict['L_total']:.4f}"
                f", gate={loss_dict.get('L_gate', 0.0):.4f}"
            )

        snapshot = {}
        tracker = self.infra.pab_tracker
        if tracker is not None:
            interval = self.config.get("pab", {}).get("checkpoint_interval", 50)
            if self.step % interval == 0:
                snapshot = self._compute_validation_snapshot()
                self._last_validation_snapshot = snapshot
                if snapshot:
                    self._last_tier_accuracies = {
                        "tier1": float(snapshot.get("tier1_accuracy", 0.0)),
                        "tier2": float(snapshot.get("validate_strict_rate", 0.0)),
                        "tier3": float(snapshot.get("compile_strict_rate", 0.0)),
                    }
                    self._last_domain_accuracies = dict(snapshot.get("domain_accuracies", {}))
                    self._last_action_accuracies = dict(snapshot.get("action_accuracies", {}))

        return record_pab_checkpoint(
            self.step,
            loss_dict,
            self.config,
            self.infra,
            getattr(self, "_last_bridge_features", None),
            self.pipeline.decoder,
            val_loss=snapshot.get("val_loss") if snapshot else None,
            tier_accuracies=self._last_tier_accuracies,
            domain_accuracies=self._last_domain_accuracies,
            action_accuracies=self._last_action_accuracies,
        )

    def _finalize_training(self, all_losses, epoch, elapsed, log_path) -> dict:
        """Save checkpoint, write log, finalize PAB profile, return summary."""
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

        result = {
            "status": "complete",
            "steps": self.step,
            "epochs": epoch,
            "elapsed_s": round(elapsed, 1),
            "final_loss": all_losses[-1] if all_losses else {},
            "checkpoint": str(ckpt_path),
        }
        if self._last_validation_snapshot:
            result["final_validation_snapshot"] = self._last_validation_snapshot

        tracker = self.infra.pab_tracker
        if tracker is not None:
            profile = tracker.finalize()
            pab_cfg = self.config.get("pab", {})
            if pab_cfg.get("save_profiles", True):
                profile_path = self.run_dir / f"{self.run_id}_pab_profile.json"
                profile.save(profile_path)
                result["pab_profile"] = str(profile_path)
            print(f"  PAB regime: {profile.summary.stability_regime}")
            print(f"  PAB stability_mean: {profile.summary.stability_mean:.4f}")

        return result

    def train(self, dry_run: bool = False) -> dict:
        """Main training loop."""
        import time

        from torch.utils.data import DataLoader

        self.setup()

        max_iters = self.config["training"]["max_iterations"]
        safety = self.config["safety"]
        log_path = self.run_dir / f"{self.run_id}{self.config['logging']['training_log_suffix']}"

        bs = self.config["training"]["batch_size"]
        print(f"Training run: {self.run_id}")
        print(f"  Device: {self.device}, Max iters: {max_iters}, Batch: {bs}")
        print(f"  Train examples: {len(self.infra.dataset)}")
        if dry_run:
            print("  [DRY RUN] Will run 1 batch only.")

        start = time.time()
        all_losses: list[dict[str, float]] = []

        loader = DataLoader(
            self.infra.dataset,
            batch_size=bs,
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
    parser = argparse.ArgumentParser(description="Balanced Sashimi training loop")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
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
