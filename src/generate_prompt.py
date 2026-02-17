"""
ShortcutForge Prompt Builder: Constructs LLM prompts for DSL generation.

Builds a system prompt with:
  - Full ShortcutDSL grammar (Lark LALR)
  - DSL rules and conventions
  - Top N most-used actions from the catalog (Tier 1)
  - Dynamically selected actions relevant to the user prompt (Tier 2)
  - Few-shot examples (linear, branching, menu)

Also provides retry message construction for the orchestrator's feedback loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_REFS_DIR = _SCRIPT_DIR.parent / "references"
_GRAMMAR_PATH = _REFS_DIR / "shortcutdsl.lark"
_CATALOG_PATH = _REFS_DIR / "action_catalog.json"

# ── Cached data ────────────────────────────────────────────────────────

_grammar_text: str | None = None
_catalog: dict | None = None
_top_actions_text: str | None = None
_reverse_canonical: dict[str, str] | None = None  # full_id → short_name


def _load_grammar() -> str:
    global _grammar_text
    if _grammar_text is None:
        _grammar_text = _GRAMMAR_PATH.read_text()
    return _grammar_text


def _load_catalog() -> dict:
    global _catalog
    if _catalog is None:
        with open(_CATALOG_PATH) as f:
            _catalog = json.load(f)
    return _catalog


def _get_reverse_canonical() -> dict[str, str]:
    """Build a reverse map: full identifier → short name."""
    global _reverse_canonical
    if _reverse_canonical is None:
        catalog = _load_catalog()
        cmap = catalog.get("_meta", {}).get("canonical_map", {})
        _reverse_canonical = {v: k for k, v in cmap.items()}
    return _reverse_canonical


def _short_name(identifier: str) -> str:
    """Get short name for an action identifier."""
    rev = _get_reverse_canonical()
    if identifier in rev:
        return rev[identifier]
    # Fall back: strip is.workflow.actions. prefix
    if identifier.startswith("is.workflow.actions."):
        return identifier[len("is.workflow.actions."):]
    return identifier


# ── Tier 1: Top N Actions ─────────────────────────────────────────────

# Internal/system actions that the LLM should NOT generate directly
_SYSTEM_ACTIONS = frozenset({
    "conditional", "choosefrommenu", "repeat.each", "repeat.count",
    "nothing", "setvariable", "getvariable", "appendvariable",
})

_SKIP_PARAMS = frozenset({
    "UUID", "CustomOutputName", "GroupingIdentifier", "WFControlFlowMode",
    "WFMenuItems", "WFMenuItemTitle", "WFMenuItemAttributedTitle",
    "WFMenuLegacyCancelBehavior",
})


def _build_top_actions(n: int = 50) -> str:
    """Build a text block of the top N most-used actions for the system prompt."""
    global _top_actions_text
    if _top_actions_text is not None:
        return _top_actions_text

    catalog = _load_catalog()
    all_actions = catalog.get("actions", {})

    scored: list[tuple[str, int, dict]] = []
    for ident, entry in all_actions.items():
        if not isinstance(entry, dict):
            continue
        short = _short_name(ident)
        if short in _SYSTEM_ACTIONS:
            continue
        usage = entry.get("usage_count", 0)
        scored.append((ident, usage, entry))

    scored.sort(key=lambda x: x[1], reverse=True)

    lines = []
    for ident, usage, entry in scored[:n]:
        short = _short_name(ident)
        desc = entry.get("description", "").strip()
        if not desc:
            desc = entry.get("name", short)
        # Truncate long descriptions
        if len(desc) > 80:
            desc = desc[:77] + "..."

        params = entry.get("observed_params", {})
        param_names = [p for p in params if p not in _SKIP_PARAMS]
        params_str = ", ".join(param_names[:8])
        if len(param_names) > 8:
            params_str += f" (+{len(param_names) - 8})"

        lines.append(f"  {short}: {desc}")
        if params_str:
            lines.append(f"    params: {params_str}")

    _top_actions_text = "\n".join(lines)
    return _top_actions_text


# ── Tier 2: Dynamic Action Selection ──────────────────────────────────

# Maps common user intent words to relevant action short names
INTENT_MAP: dict[str, list[str]] = {
    # Communication
    "message": ["sendmessage"],
    "text": ["sendmessage", "gettext"],
    "sms": ["sendmessage"],
    "email": ["sendemail"],
    "mail": ["sendemail"],
    # Media
    "photo": ["selectphoto", "savetocameraroll", "takephoto"],
    "camera": ["takephoto", "selectphoto"],
    "image": ["selectphoto", "savetocameraroll", "overlaytext"],
    "video": ["selectphoto", "takevideo"],
    "music": ["playmusic", "searchitunes", "getcurrentlyplayingsong"],
    "song": ["playmusic", "searchitunes", "getcurrentlyplayingsong"],
    # Productivity
    "calendar": ["addnewevent", "filter.calendarevents"],
    "event": ["addnewevent", "filter.calendarevents"],
    "reminder": ["addnewreminder", "filter.reminders"],
    "note": ["shownote", "appendtonote", "createnote"],
    "timer": ["delay", "date", "adjustdate"],
    "alarm": ["com.apple.mobiletimer-framework.MobileTimerIntents.MTGetAlarmsIntent"],
    "clipboard": ["getclipboard", "setclipboard"],
    "file": ["documentpicker.open", "documentpicker.save", "file.append"],
    "save": ["documentpicker.save", "savetocameraroll"],
    # Web & Data
    "weather": ["weather.currentconditions"],
    "web": ["url", "downloadurl", "openurl"],
    "api": ["url", "downloadurl", "detect.dictionary", "getvalueforkey"],
    "http": ["url", "downloadurl"],
    "json": ["detect.dictionary", "getvalueforkey", "dictionary"],
    "url": ["url", "openurl", "detect.link"],
    "download": ["downloadurl"],
    "search": ["searchitunes", "searchapp"],
    # Health
    "health": ["health.quantity.log"],
    "caffeine": ["health.quantity.log"],
    "water": ["health.quantity.log"],
    "workout": ["health.quantity.log"],
    # Device
    "brightness": ["setbrightness"],
    "volume": ["setvolume"],
    "wifi": ["wifi.set"],
    "bluetooth": ["bluetooth.set"],
    "flashlight": ["flashlight"],
    "wallpaper": ["wallpaper.set"],
    "airplane": ["airplanemode.set"],
    "focus": ["setfocusmode"],
    "dnd": ["setfocusmode"],
    # Data Processing
    "regex": ["text.replace", "text.match"],
    "split": ["text.split"],
    "replace": ["text.replace"],
    "count": ["count"],
    "filter": ["filter.content"],
    "sort": ["filter.content"],
    "random": ["randomnumber"],
    "math": ["math", "number"],
    "calculate": ["math", "number"],
    # Location
    "location": ["location", "getmapslink"],
    "map": ["getmapslink", "searchlocalmaps"],
    "directions": ["getdirectionsaction"],
    "address": ["location", "getmapslink"],
    # Misc
    "qr": ["generatebarcode"],
    "barcode": ["generatebarcode", "scanbarcode"],
    "share": ["share"],
    "speak": ["speaktext"],
    "vibrate": ["vibrate"],
    "notification": ["notification", "alert"],
    "alert": ["alert", "notification"],
    "shortcut": ["runworkflow"],
    "encode": ["base64encode"],
    "decode": ["base64encode"],
    "hash": ["hash"],
}


def select_relevant_actions(prompt: str, top_n: int = 20) -> str:
    """Select catalog actions relevant to the user's prompt.

    Uses keyword matching against INTENT_MAP and action descriptions.
    Returns a formatted text block of relevant actions.
    """
    catalog = _load_catalog()
    all_actions = catalog.get("actions", {})

    # Tokenize prompt into lowercase words
    words = set(re.findall(r"[a-z]+", prompt.lower()))

    # Collect relevant action short names from INTENT_MAP
    relevant_shorts: set[str] = set()
    for word in words:
        if word in INTENT_MAP:
            relevant_shorts.update(INTENT_MAP[word])
        # Also check partial matches (e.g., "photos" matches "photo")
        for intent_word, action_list in INTENT_MAP.items():
            if intent_word in word or word in intent_word:
                relevant_shorts.update(action_list)

    # Also match against action names and descriptions
    for ident, entry in all_actions.items():
        if not isinstance(entry, dict):
            continue
        short = _short_name(ident)
        if short in _SYSTEM_ACTIONS:
            continue
        desc = (entry.get("description", "") + " " + entry.get("name", "")).lower()
        for word in words:
            if len(word) >= 4 and word in desc:
                relevant_shorts.add(short)
                break

    # Remove actions already in Tier 1 top 50
    # (We don't want duplicates — Tier 1 is already in the prompt)
    _build_top_actions()  # ensure top actions are cached
    top_50_catalog = _load_catalog().get("actions", {})
    scored_top = []
    for ident, entry in top_50_catalog.items():
        if not isinstance(entry, dict):
            continue
        short = _short_name(ident)
        if short in _SYSTEM_ACTIONS:
            continue
        scored_top.append((short, entry.get("usage_count", 0)))
    scored_top.sort(key=lambda x: x[1], reverse=True)
    top_50_shorts = {s for s, _ in scored_top[:50]}

    relevant_shorts -= top_50_shorts

    if not relevant_shorts:
        return ""

    # Build text for relevant actions
    lines = []
    for short in sorted(relevant_shorts)[:top_n]:
        # Find the full identifier
        cmap = catalog.get("_meta", {}).get("canonical_map", {})
        full_id = cmap.get(short, short)
        entry = all_actions.get(full_id, {})
        if not entry:
            # Try direct lookup
            for ident, e in all_actions.items():
                if _short_name(ident) == short:
                    entry = e
                    break
        if not entry:
            continue

        desc = entry.get("description", "").strip()
        if not desc:
            desc = entry.get("name", short)
        if len(desc) > 80:
            desc = desc[:77] + "..."

        params = entry.get("observed_params", {})
        param_names = [p for p in params if p not in _SKIP_PARAMS]
        params_str = ", ".join(param_names[:6])

        lines.append(f"  {short}: {desc}")
        if params_str:
            lines.append(f"    params: {params_str}")

    return "\n".join(lines)


# ── DSL Rules ──────────────────────────────────────────────────────────

_DSL_RULES = """\
## DSL Rules

1. Every shortcut starts with: SHORTCUT "Name Here"
2. Actions: ACTION <name> <param>=<value> [<param>=<value> ...]
3. ALWAYS use the shortest action name. Use "comment" not "is.workflow.actions.comment".
   Use "downloadurl" not "is.workflow.actions.downloadurl". The pipeline resolves short names automatically.
4. String values are double-quoted: "hello world"
5. Variables: SET $VarName = <value> to store, then $VarName to reference
6. @prev refers to the output of the immediately preceding action
7. Use SET $var = @prev when you need the value more than once or after other actions
8. Interpolated strings use backticks: `Hello {$Name}, today is {@date}`
9. Control flow blocks MUST be properly closed:
   - IF ... ENDIF (with optional ELSE)
   - MENU "prompt" ... CASE "label" ... ENDMENU
   - REPEAT N ... ENDREPEAT
   - FOREACH @prev ... ENDFOREACH (use @item and @index inside)
10. IF conditions: has_any_value, does_not_have_any_value, equals_number,
    is_greater_than, is_less_than, equals_string, not_equal_string,
    contains, does_not_contain, is_before
11. Dictionaries: {"key": "value", "count": 42}
12. Lists: ["item1", "item2"]
13. Quantities: QTY(7, "days") or QTY(30, "min")
14. HTTP headers: HEADERS {"Content-Type": "application/json"}
15. Parameter names MUST match the catalog exactly. Never invent parameter names.
16. Do NOT use conditional, choosefrommenu, repeat.each, repeat.count, setvariable,
    getvariable, or appendvariable as ACTION names — use the DSL keywords instead:
    IF/ELSE/ENDIF, MENU/CASE/ENDMENU, REPEAT/ENDREPEAT, FOREACH/ENDFOREACH, SET.
17. For showing text to the user, use: ACTION showresult Text="message" or Text=`interpolated {$var}`
18. For notifications: ACTION notification WFNotificationActionTitle="Title" WFNotificationActionBody="Body"
19. For asking the user: ACTION ask WFAskActionPrompt="Question?" WFInputType="Text"
20. For delays/timers: ACTION delay WFDelayTime=300 (seconds)
21. Every shortcut MUST end with ENDSHORTCUT on its own line.
"""

# ── Few-Shot Examples ──────────────────────────────────────────────────

_EXAMPLES = """\
## Examples

### Example 1: Simple API Pipeline

User: "Fetch the current weather for Seattle and show me the temperature"

```
SHORTCUT "Weather Check"

ACTION comment WFCommentActionText="Fetch weather data"
ACTION url WFURLActionURL="https://api.weather.example.com/current?city=Seattle"
ACTION downloadurl WFHTTPMethod="GET"
ACTION detect.dictionary
SET $Weather = @prev
ACTION getvalueforkey WFDictionaryKey="temperature" WFInput=$Weather
ACTION showresult Text=`Current temperature: {@prev}°F`
ENDSHORTCUT
```

### Example 2: Conditional Branching

User: "Ask the user for their name. If they provide one, greet them. Otherwise show an error."

```
SHORTCUT "Greeter"

ACTION ask WFAskActionPrompt="What is your name?" WFInputType="Text"
SET $Name = @prev
IF $Name has_any_value
  ACTION showresult Text=`Hello, {$Name}! Welcome!`
ELSE
  ACTION alert WFAlertActionMessage="You didn't enter a name!" WFAlertActionTitle="Error"
ENDIF
ENDSHORTCUT
```

### Example 3: Menu with Variables

User: "Create a shortcut with a menu to choose between 5, 10, or 15 minute timers"

```
SHORTCUT "Quick Timer"

MENU "How long?"
CASE "5 minutes"
  SET $Seconds = 300
CASE "10 minutes"
  SET $Seconds = 600
CASE "15 minutes"
  SET $Seconds = 900
ENDMENU
ACTION showresult Text=`Timer set for {$Seconds} seconds`
ACTION delay WFDelayTime=$Seconds
ACTION notification WFNotificationActionTitle="Timer Done!" WFNotificationActionBody="Your timer has finished."
ENDSHORTCUT
```
"""

# ── Snippet Context ────────────────────────────────────────────────────

def build_snippet_context(
    prompt: str,
    registry_path: Path | None = None,
) -> str:
    """Build a text block of relevant DSL snippet examples for the user message.

    Queries the snippet registry for patterns that match the user's prompt
    and formats them as canonical DSL examples the LLM can reference.

    Args:
        prompt: The user's natural language prompt.
        registry_path: Path to snippet_registry.json. Defaults to
            references/snippet_registry.json.

    Returns:
        Formatted string with canonical DSL examples, or empty string
        if no registry exists or no matches found.
    """
    try:
        from snippet_extractor import query_snippets
    except ImportError:
        return ""

    if registry_path is None:
        registry_path = _REFS_DIR / "snippet_registry.json"

    if not registry_path.exists():
        return ""

    try:
        results = query_snippets(prompt, registry_path=registry_path, top_k=3)
    except Exception:
        return ""

    if not results:
        return ""

    lines: list[str] = []
    for i, snip in enumerate(results, 1):
        desc = snip.get("description", "")
        canonical = snip.get("canonical_dsl", "")
        if not canonical:
            continue
        lines.append(f"### Pattern {i}: {desc}")
        lines.append(f"```")
        lines.append(canonical)
        lines.append(f"```")
        lines.append("")

    return "\n".join(lines).rstrip()


# ── Execution Plan Context ─────────────────────────────────────────────


def build_plan_context(plan: Any) -> str:
    """Format an ExecutionPlan as structured context for the generation prompt.

    Takes an ExecutionPlan object (from execution_planner) and formats it
    as a text block suitable for appending to the system prompt addendum.

    Degrades gracefully: returns empty string if the plan has no steps
    or if the execution_planner module is not available.

    Args:
        plan: An ExecutionPlan object with archetype, steps, etc.

    Returns:
        A formatted multi-line string, or empty string if no plan context.
    """
    try:
        if plan is None or not hasattr(plan, "steps") or not plan.steps:
            return ""

        lines: list[str] = []
        lines.append("## Execution Plan")
        lines.append(f"Archetype: {plan.archetype}")
        lines.append("Steps:")
        for i, step in enumerate(plan.steps, 1):
            actions_str = (
                ", ".join(step.candidate_actions)
                if step.candidate_actions
                else "(none)"
            )
            lines.append(
                f"  {i}. {step.description} "
                f"-> likely actions: {actions_str}"
            )

        return "\n".join(lines)
    except Exception:
        return ""


# ── Public API ─────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build the static system prompt for DSL generation.

    Contains: role/task, full grammar, DSL rules, top 50 actions, few-shot examples.
    """
    grammar = _load_grammar()
    top_actions = _build_top_actions(50)

    return f"""\
You are ShortcutForge, an Apple Shortcuts generator. Given a natural language \
description, you produce valid ShortcutDSL code that compiles to a working \
Apple Shortcut (.shortcut file).

Output ONLY the DSL code. No markdown fences, no explanation, no preamble, \
no commentary. Start with SHORTCUT and end with ENDSHORTCUT.

Be succinct: use the shortest valid action names (e.g. "downloadurl" not \
"is.workflow.actions.downloadurl"). The pipeline resolves short names to \
their full identifiers automatically. Focus on correctness and brevity.

## ShortcutDSL Grammar (Lark LALR)

```
{grammar}
```

{_DSL_RULES}

## Available Actions (most common, sorted by usage)

{top_actions}

{_EXAMPLES}"""


def build_user_message(
    prompt: str,
    domain_context: str = "",
    domain_actions: str = "",
    include_snippets: bool = False,
    snippet_context: str = "",
) -> str:
    """Build the user message with dynamic action context.

    Includes:
      - Domain profile context (if a specialized domain was detected)
      - Domain-relevant actions (from domain profile, not in Tier 1/2)
      - Snippet context (retrieved DSL patterns relevant to the prompt)
      - Tier 2 actions relevant to the prompt
      - The prompt itself

    Args:
        prompt: The user's natural language prompt.
        domain_context: Prompt context from domain profile (e.g., HealthKit guidance).
        domain_actions: Formatted relevant actions from domain profile.
        include_snippets: Whether to include snippet context in the message.
        snippet_context: Formatted DSL patterns from snippet registry.
    """
    tier2 = select_relevant_actions(prompt)

    parts = []
    if domain_context:
        parts.append(f"Domain-specific guidance:\n{domain_context}\n")
    if domain_actions:
        parts.append(f"Domain-relevant actions:\n{domain_actions}\n")
    if include_snippets and snippet_context:
        parts.append(f"Relevant DSL patterns:\n{snippet_context}\n")
    if tier2:
        parts.append(f"Additional relevant actions for this request:\n{tier2}\n")
    parts.append(f"Build this shortcut: {prompt}")
    return "\n".join(parts)


def build_retry_message(dsl_text: str, errors: list[str]) -> str:
    """Build a retry message with the failed DSL and structured errors.

    Used by the orchestrator when parse or validation fails.
    The message is appended to the conversation as a user turn.
    """
    error_list = "\n".join(f"  - {e}" for e in errors)
    return (
        f"Your previous DSL output had the following errors:\n"
        f"{error_list}\n\n"
        f"Here was your output:\n"
        f"```\n{dsl_text}\n```\n\n"
        f"Fix these errors and output the corrected DSL. "
        f"Output ONLY the DSL code, starting with SHORTCUT and ending with ENDSHORTCUT."
    )


# ── CLI test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Set a 5-minute timer and notify me"

    system = build_system_prompt()
    user = build_user_message(prompt)

    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(system)
    print()
    print("=" * 70)
    print(f"USER MESSAGE (prompt: {prompt!r})")
    print("=" * 70)
    print(user)
    print()

    # Rough token estimate (1 token ≈ 4 chars)
    total_chars = len(system) + len(user)
    print(f"Estimated tokens: ~{total_chars // 4:,} (system: ~{len(system) // 4:,}, user: ~{len(user) // 4:,})")
