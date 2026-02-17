"""
Local DSL generation using fine-tuned MLX model + Outlines grammar constraint.

Provides LocalDSLGenerator that integrates with the Orchestrator as a
pluggable GeneratorBackend.

Usage:
    from inference import LocalDSLGenerator

    gen = LocalDSLGenerator(model_path="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                            adapter_path="./checkpoints/adapter")
    dsl = gen.generate(system_prompt, messages, max_tokens=4096)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Grammar path relative to project root
_PROJECT_ROOT = _SCRIPT_DIR.parent
_GRAMMAR_PATH = _PROJECT_ROOT / "references" / "shortcutdsl_outlines.lark"

# Default wall-clock timeout for generation (seconds)
DEFAULT_TIMEOUT_S = 90


@dataclass
class GenerationMeta:
    """Metadata about a generation run."""
    timed_out: bool = False
    gen_time_s: float = 0.0
    tokens_generated: int = 0
    early_stopped: bool = False  # True if stopped on ENDSHORTCUT


def generate_with_timeout(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, GenerationMeta]:
    """Generate text with wall-clock timeout using mlx_lm.stream_generate.

    Uses streaming to check wall-clock time after each token. Also
    performs early stopping when ENDSHORTCUT is detected (the natural
    end-of-shortcut marker).

    Args:
        model: Raw MLX model (NOT outlines-wrapped).
        tokenizer: MLX tokenizer.
        prompt: Formatted prompt string.
        max_tokens: Maximum tokens to generate.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        (generated_text, GenerationMeta) tuple.
    """
    import mlx_lm

    accumulated = ""
    meta = GenerationMeta()
    t0 = time.monotonic()

    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        # stream_generate yields GenerationResponse with .text per-token.
        # Accumulate tokens into the full generated text.
        accumulated += response.text
        meta.tokens_generated += 1

        # Wall-clock timeout check
        elapsed = time.monotonic() - t0
        if elapsed > timeout_s:
            meta.timed_out = True
            meta.gen_time_s = elapsed
            return accumulated, meta

        # Early stop on ENDSHORTCUT — the generation is complete
        # Check periodically (every 10 tokens to avoid overhead)
        if meta.tokens_generated % 10 == 0:
            if "ENDSHORTCUT" in accumulated:
                # Find ENDSHORTCUT and trim everything after the first newline following it
                idx = accumulated.index("ENDSHORTCUT")
                end_idx = accumulated.find("\n", idx)
                if end_idx != -1:
                    result = accumulated[:end_idx + 1]
                else:
                    result = accumulated[:idx + len("ENDSHORTCUT")] + "\n"
                meta.early_stopped = True
                meta.gen_time_s = time.monotonic() - t0
                return result, meta

    # Normal completion (max_tokens reached or EOS)
    meta.gen_time_s = time.monotonic() - t0
    result = accumulated

    # Final check for ENDSHORTCUT — trim trailing content if present
    if "ENDSHORTCUT" in result:
        idx = result.index("ENDSHORTCUT")
        end_idx = result.find("\n", idx)
        if end_idx != -1:
            result = result[:end_idx + 1]
        else:
            result = result[:idx + len("ENDSHORTCUT")] + "\n"
        meta.early_stopped = True

    return result, meta


class LocalDSLGenerator:
    """Local MLX-based DSL generator with optional Outlines grammar constraint.

    Loads a fine-tuned (or base) Llama model via MLX-LM and uses Outlines
    CFG grammar to constrain generation to syntactically valid DSL.

    Supports two generation modes:
      - Unconstrained (default): Uses mlx_lm.stream_generate with wall-clock
        timeout and early ENDSHORTCUT stopping.
      - Grammar-constrained: Uses Outlines CFG for syntactically guaranteed
        output (no streaming timeout — Outlines doesn't support it).
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        use_grammar: bool = True,
        grammar_path: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        chat_template: str = "llama3",
    ):
        """Initialize the local generator.

        Args:
            model_path: Path or HuggingFace repo for the base model.
            adapter_path: Path to LoRA adapter directory (optional).
            use_grammar: Whether to enable grammar-constrained generation.
            grammar_path: Path to Lark grammar file (default: shortcutdsl_outlines.lark).
            timeout_s: Wall-clock timeout in seconds for unconstrained generation.
            chat_template: Chat template format ("llama3" or "chatml").
        """
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._use_grammar = use_grammar
        self._grammar_path = grammar_path or str(_GRAMMAR_PATH)
        self._timeout_s = timeout_s
        self._chat_template = chat_template

        # Lazy-loaded
        self._raw_model = None       # Raw MLX model (for streaming)
        self._raw_tokenizer = None   # Raw tokenizer (for streaming)
        self._outlines_model = None  # Outlines-wrapped model (for grammar)
        self._cfg = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load model and grammar on first use."""
        if self._loaded:
            return

        import mlx_lm
        import outlines

        # Load model (with optional adapter)
        load_kwargs = {}
        if self._adapter_path:
            load_kwargs["adapter_path"] = self._adapter_path

        model, tokenizer = mlx_lm.load(self._model_path, **load_kwargs)

        # Store raw model + tokenizer for streaming generation
        self._raw_model = model
        self._raw_tokenizer = tokenizer

        # Wrap in Outlines for grammar-constrained generation
        self._outlines_model = outlines.from_mlxlm(model, tokenizer)

        # Load grammar
        if self._use_grammar:
            from outlines.types import CFG
            grammar_text = Path(self._grammar_path).read_text()
            self._cfg = CFG(grammar_text)

        self._loaded = True

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
        timeout_s: float | None = None,
        **kwargs,
    ) -> str:
        """Generate DSL from a prompt using the local model.

        Args:
            system_prompt: System prompt text.
            messages: List of message dicts with "role" and "content".
            max_tokens: Maximum tokens to generate.
            timeout_s: Wall-clock timeout override (default: instance setting).
            **kwargs: Ignored (for interface compatibility with ClaudeBackend).

        Returns:
            Raw generated text (DSL).
        """
        self._ensure_loaded()

        # Format as Llama 3 chat template
        formatted = self._format_prompt(system_prompt, messages)

        if self._use_grammar and self._cfg is not None:
            # Grammar-constrained path — no streaming timeout available
            model = self._outlines_model
            assert model is not None, "Outlines model not loaded"
            result = model(formatted, self._cfg, max_tokens=max_tokens)
            return result
        else:
            # Unconstrained path — use streaming with timeout
            assert self._raw_model is not None, "Model not loaded"
            assert self._raw_tokenizer is not None, "Tokenizer not loaded"
            effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
            text, _meta = generate_with_timeout(
                self._raw_model,
                self._raw_tokenizer,
                formatted,
                max_tokens=max_tokens,
                timeout_s=effective_timeout,
            )
            return text

    def generate_with_meta(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
        timeout_s: float | None = None,
    ) -> tuple[str, GenerationMeta]:
        """Generate DSL and return metadata (timeout status, timing, etc.).

        Same as generate() but returns GenerationMeta alongside the text.
        Only available for unconstrained generation; grammar mode returns
        default meta.
        """
        self._ensure_loaded()
        formatted = self._format_prompt(system_prompt, messages)

        if self._use_grammar and self._cfg is not None:
            model = self._outlines_model
            assert model is not None, "Outlines model not loaded"
            t0 = time.monotonic()
            result = model(formatted, self._cfg, max_tokens=max_tokens)
            meta = GenerationMeta(gen_time_s=time.monotonic() - t0)
            return result, meta
        else:
            assert self._raw_model is not None, "Model not loaded"
            assert self._raw_tokenizer is not None, "Tokenizer not loaded"
            effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
            return generate_with_timeout(
                self._raw_model,
                self._raw_tokenizer,
                formatted,
                max_tokens=max_tokens,
                timeout_s=effective_timeout,
            )

    def _format_prompt(self, system_prompt: str, messages: list[dict]) -> str:
        """Format messages using the configured chat template."""
        if self._chat_template == "chatml":
            return self._format_chatml(system_prompt, messages)
        return self._format_llama3(system_prompt, messages)

    def _format_chatml(self, system_prompt: str, messages: list[dict]) -> str:
        """Format messages as ChatML template (Qwen, etc.)."""
        parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _format_llama3(self, system_prompt: str, messages: list[dict]) -> str:
        """Format messages as Llama 3 chat template."""
        parts = ["<|begin_of_text|>"]

        # System message
        parts.append("<|start_header_id|>system<|end_header_id|>\n")
        parts.append(system_prompt)
        parts.append("<|eot_id|>")

        # User/assistant messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n")
            parts.append(content)
            parts.append("<|eot_id|>")

        # Prompt for assistant response
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return "".join(parts)

    @staticmethod
    def is_available() -> bool:
        """Check if MLX and Outlines are installed."""
        try:
            import mlx_lm  # noqa: F401
            import outlines  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def engine_name(self) -> str:
        return "local"
