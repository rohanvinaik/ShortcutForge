"""
Dynamic Token Budgeting for ShortcutForge.

Estimates appropriate max_tokens budget based on prompt complexity,
preventing token overflow failures (exhausting max_tokens mid-generation)
while reducing latency for simple prompts.

Complexity buckets:
  Simple       (512 tokens):  Short prompts, single actions, no complex signals
  Medium       (1024 tokens): Default — covers P90 of training data
  Complex      (2048 tokens): Multi-step, conditional, API-heavy prompts
  Very Complex (4096 tokens): Highly complex workflows, many steps, long prompts

Also provides overflow detection and progressive budget escalation
for retry before parse/validate retry.

Usage:
    from token_budget import estimate_budget, detect_overflow, next_budget

    budget = estimate_budget(prompt)
    raw = generate(..., max_tokens=budget.max_tokens)

    if detect_overflow(raw, budget, "ENDSHORTCUT" in raw):
        new_budget = next_budget(budget.max_tokens)
        if new_budget is not None:
            raw = generate(..., max_tokens=new_budget)
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Budget tiers for progressive escalation ───────────────────────

BUDGET_TIERS = [512, 1024, 2048, 4096]


# ── Complex/simple signal keywords ────────────────────────────────

_COMPLEX_SIGNALS = frozenset(
    {
        "menu",
        "if",
        "repeat",
        "loop",
        "dictionary",
        "api",
        "json",
        "multiple",
        "list",
        "batch",
        "each",
        "every",
        "for each",
        "conditional",
        "check",
        "compare",
        "filter",
        "iterate",
        "convert",
        "scan",
        "qr",
        "document",
        "collage",
        "pomodoro",
        "cycle",
        "routine",
        "manager",
        "logger",
        "quiz",
        # Health/HealthKit signals (Phase 1: domain-aware budgeting)
        "health",
        "healthkit",
        "nutrient",
        "supplement",
        "vitamin",
        "mineral",
        "caffeine",
        "workout",
        "fitness",
        "apple health",
    }
)

_SIMPLE_SIGNALS = frozenset(
    {
        "open",
        "set",
        "toggle",
        "turn",
        "play",
        "show",
        "get",
        "launch",
        "start",
        "stop",
        "enable",
        "disable",
    }
)


# ── Token Budget Dataclass ────────────────────────────────────────


@dataclass
class TokenBudget:
    """Estimated token budget for a prompt."""

    max_tokens: int
    complexity: str  # "simple", "medium", "complex", "very_complex"
    word_count: int
    complex_signal_count: int
    simple_signal_count: int
    prompt_char_len: int


# ── Budget Estimation ─────────────────────────────────────────────


def estimate_budget(prompt: str) -> TokenBudget:
    """Estimate appropriate max_tokens for a prompt based on complexity.

    Features analyzed:
      - word_count: number of words in the prompt
      - complex_signals: count of keywords indicating multi-step/conditional logic
      - simple_signals: count of keywords indicating single actions
      - prompt_char_len: raw character length

    Bucket assignment:
      Simple:       word_count <= 12, no complex signals, has simple signal → 512
      Very Complex: complex_signals >= 4, OR word_count > 80, OR char_len > 500 → 4096
      Complex:      complex_signals >= 2, OR word_count > 40, OR char_len > 300 → 2048
      Medium:       everything else → 1024
    """
    words = prompt.lower().split()
    word_count = len(words)
    prompt_lower = prompt.lower()

    # Count signal matches
    complex_count = sum(1 for sig in _COMPLEX_SIGNALS if sig in prompt_lower)
    simple_count = sum(1 for sig in _SIMPLE_SIGNALS if sig in prompt_lower)

    # Bucket assignment (check most-complex first)
    if complex_count >= 4 or word_count > 80 or len(prompt) > 500:
        complexity = "very_complex"
        max_tokens = 4096
    elif word_count <= 12 and complex_count == 0 and simple_count > 0:
        complexity = "simple"
        max_tokens = 512
    elif complex_count >= 2 or word_count > 40 or len(prompt) > 300:
        complexity = "complex"
        max_tokens = 2048
    else:
        complexity = "medium"
        max_tokens = 1024

    return TokenBudget(
        max_tokens=max_tokens,
        complexity=complexity,
        word_count=word_count,
        complex_signal_count=complex_count,
        simple_signal_count=simple_count,
        prompt_char_len=len(prompt),
    )


# ── Progressive Budget Escalation ─────────────────────────────────


def next_budget(current: int) -> int | None:
    """Get the next budget tier above the current one.

    Returns None if already at max tier.

    Examples:
        next_budget(512)  → 1024
        next_budget(1024) → 2048
        next_budget(2048) → 4096
        next_budget(4096) → None
    """
    for tier in BUDGET_TIERS:
        if tier > current:
            return tier
    return None


# ── Overflow Detection ────────────────────────────────────────────


def detect_overflow(
    raw_output: str,
    budget: TokenBudget,
    has_endshortcut: bool,
) -> bool:
    """Detect if generation likely hit the token budget limit.

    Returns True if:
      - Output length is near the budget limit (>= 85% of max_tokens)
      - No ENDSHORTCUT terminator present
      - Output doesn't end with a valid structural closer

    This detects truncated output from token exhaustion, which should
    trigger a budget retry (2x) before parse/validate retry.
    """
    # Approximate tokens (~3.5 chars per token for DSL)
    approx_tokens = len(raw_output) / 3.5
    near_budget = approx_tokens >= (budget.max_tokens * 0.85)

    no_terminator = not has_endshortcut

    valid_enders = ("ENDIF", "ENDMENU", "ENDFOREACH", "ENDREPEAT", "ENDSHORTCUT")
    stripped_end = raw_output.rstrip()
    no_valid_end = not any(stripped_end.endswith(e) for e in valid_enders)

    return near_budget and no_terminator and no_valid_end


# ── Calibration (development tool) ───────────────────────────────


def calibrate(train_file: str) -> None:
    """Print bucket statistics from training data for heuristic validation.

    Usage:
        python -c "from token_budget import calibrate; calibrate('training_data/shortcutdsl_train.jsonl')"
    """
    import json

    budgets = {"simple": [], "medium": [], "complex": [], "very_complex": []}
    dsl_tokens = []

    with open(train_file) as f:
        for line in f:
            example = json.loads(line)
            prompt = example["messages"][1]["content"]
            dsl = example["messages"][2]["content"]

            budget = estimate_budget(prompt)
            dsl_len = len(dsl) / 3.5  # approx tokens

            budgets[budget.complexity].append(dsl_len)
            dsl_tokens.append(dsl_len)

    dsl_tokens.sort()
    total = len(dsl_tokens)

    print(f"\nToken Budget Calibration ({total} training examples)")
    print(f"{'=' * 60}")
    print("Overall DSL token stats:")
    print(f"  Median: {dsl_tokens[total // 2]:.0f}")
    print(f"  P90:    {dsl_tokens[int(total * 0.9)]:.0f}")
    print(f"  P95:    {dsl_tokens[int(total * 0.95)]:.0f}")
    print(f"  P99:    {dsl_tokens[int(total * 0.99)]:.0f}")
    print(f"  Max:    {dsl_tokens[-1]:.0f}")

    for bucket_name, tokens_list in sorted(budgets.items()):
        if not tokens_list:
            print(f"\n{bucket_name}: 0 examples")
            continue
        tokens_list.sort()
        n = len(tokens_list)
        print(f"\n{bucket_name} ({n} examples, {n / total * 100:.1f}%):")
        print(f"  Median: {tokens_list[n // 2]:.0f}")
        print(f"  P90:    {tokens_list[int(n * 0.9)]:.0f}")
        if n > 20:
            print(f"  P95:    {tokens_list[int(n * 0.95)]:.0f}")
        print(f"  Max:    {tokens_list[-1]:.0f}")

        # Check bucket fit (what % of examples fit within budget)
        budget_limit = {
            "simple": 512,
            "medium": 1024,
            "complex": 2048,
            "very_complex": 4096,
        }[bucket_name]
        fits = sum(1 for t in tokens_list if t <= budget_limit)
        print(f"  Fits in budget ({budget_limit}): {fits}/{n} ({fits / n * 100:.1f}%)")
