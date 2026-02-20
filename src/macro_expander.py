"""
ShortcutDSL Macro Expander.

Recognizes MACRO directives in DSL text and expands them into
multi-action DSL sequences using templates from macro_patterns.json.

Operates at the text level (pre-parse). Called from dsl_linter.py
as Phase 0 of the lint pipeline.

Template syntax:
  - {{param}} — substitutes the parameter value
  - {{#list}}...{{/list}} — iterates over list items ({{.}} = current item)

Macro invocation syntax:
  MACRO <name> <param>=<value> [<param>=<value> ...]

Block macros (like platform.if_ios) also have an end marker:
  MACRO platform.if_ios
    ... body ...
  ENDPLATFORM
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any  # noqa: F401

__version__ = "2.0"

# ── Paths ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_REFS_DIR = _SCRIPT_DIR.parent / "references"
_REGISTRY_PATH = _REFS_DIR / "macro_patterns.json"

# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class MacroDefinition:
    """A macro definition from the registry."""

    name: str
    description: str
    params: dict[str, dict]
    expansion_template: str
    domain: str = "general"
    tags: list[str] | None = None
    block_macro: bool = False
    end_marker: str = ""
    end_expansion: str = ""


@dataclass
class MacroExpansion:
    """Record of a macro expansion (for change tracking)."""

    line: int
    macro_name: str
    original: str
    expanded: str
    param_values: dict[str, str]


# ── Registry Loading ──────────────────────────────────────────────────

_registry: dict[str, MacroDefinition] | None = None


def _load_registry(registry_path: Path | None = None) -> dict[str, MacroDefinition]:
    """Load macro definitions from the registry JSON."""
    global _registry
    if _registry is not None and registry_path is None:
        return _registry

    path = registry_path or _REGISTRY_PATH
    if not path.exists():
        _registry = {}
        return _registry

    with open(path) as f:
        data = json.load(f)

    macros = data.get("macros", {})
    _registry = {}

    for name, defn in macros.items():
        _registry[name] = MacroDefinition(
            name=name,
            description=defn.get("description", ""),
            params=defn.get("params", {}),
            expansion_template=defn.get("expansion_template", ""),
            domain=defn.get("domain", "general"),
            tags=defn.get("tags"),
            block_macro=defn.get("block_macro", False),
            end_marker=defn.get("end_marker", ""),
            end_expansion=defn.get("end_expansion", ""),
        )

    return _registry


def get_registry(registry_path: Path | None = None) -> dict[str, MacroDefinition]:
    """Get the macro registry (public API)."""
    return _load_registry(registry_path)


def reload_registry(registry_path: Path | None = None) -> dict[str, MacroDefinition]:
    """Force reload the registry (useful for testing)."""
    global _registry
    _registry = None
    return _load_registry(registry_path)


# ── Parameter Parsing ─────────────────────────────────────────────────

# Regex to match MACRO <name> <params...>
_MACRO_LINE_RE = re.compile(
    r"^\s*MACRO\s+(\S+)\s*(.*?)\s*$",
    re.IGNORECASE,
)

# Regex to match param=value, param="value", param=$var, param=[...], param={...}
_PARAM_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"""
    r""""([^"]*?)"|"""  # "quoted string"
    r"""'([^']*?)'|"""  # 'single quoted'
    r"""(\$\w+)|"""  # $variable
    r"""(\[.*?\])|"""  # [list]
    r"""(\{.*?\})|"""  # {dict}
    r"""(\S+)"""  # bare word
    r""")""",
    re.DOTALL,
)


def _parse_macro_params(param_str: str) -> dict[str, str]:
    """Parse parameter string from a MACRO line."""
    params: dict[str, str] = {}
    for m in _PARAM_RE.finditer(param_str):
        key = m.group(1)
        # Take the first non-None captured group as the value
        value = next(
            (g for g in m.groups()[1:] if g is not None),
            "",
        )
        params[key] = value
    return params


# ── Template Rendering ────────────────────────────────────────────────


def _render_template(
    template: str, params: dict[str, str], defn: MacroDefinition
) -> str:
    """Render a macro template with parameter substitution.

    Handles:
      - {{param}} → value
      - {{#list}}...{{/list}} → repeated block for each item
      - {{?param}}...{{/param}} → conditional section (v2.0: included only if param is provided)
      - Default values from macro definition
    """
    result = template

    # Apply default values for missing params
    for param_name, param_spec in defn.params.items():
        if param_name not in params and "default" in param_spec:
            params[param_name] = param_spec["default"]

    # Handle conditional sections: {{?paramname}}...{{/paramname}}
    # Included only if the param is present in params (after defaults applied)
    cond_pattern = re.compile(r"\{\{\?(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
    for m in cond_pattern.finditer(result):
        cond_key = m.group(1)
        cond_block = m.group(2)
        if cond_key in params and params[cond_key]:
            # Include the block (with param substitution done later)
            result = result.replace(m.group(0), cond_block)
        else:
            # Omit the block entirely
            result = result.replace(m.group(0), "")

    # Handle list iteration: {{#paramname}}...{{.}}...{{/paramname}}
    list_pattern = re.compile(r"\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
    for m in list_pattern.finditer(result):
        list_key = m.group(1)
        block_template = m.group(2)
        list_value = params.get(list_key, "[]")

        # Parse the list value
        try:
            items = (
                json.loads(list_value) if isinstance(list_value, str) else list_value
            )
        except (json.JSONDecodeError, TypeError):
            items = [list_value]

        expanded_blocks = []
        for item in items:
            block = block_template.replace("{{.}}", str(item))
            expanded_blocks.append(block)

        result = result.replace(m.group(0), "".join(expanded_blocks))

    # Simple parameter substitution: {{param}} → value
    for key, value in params.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))

    return result


def validate_macro_params(
    macro_name: str,
    params: dict[str, str],
    defn: MacroDefinition,
) -> list[str]:
    """Validate macro parameters against the definition.

    Returns a list of warning/error messages. Empty list means valid.
    """
    warnings: list[str] = []

    # Check required params
    for param_name, param_spec in defn.params.items():
        if param_spec.get("required", False) and param_name not in params:
            warnings.append(
                f"Macro '{macro_name}': missing required parameter '{param_name}'"
            )

    # Check for unknown params
    known_params = set(defn.params.keys())
    for param_name in params:
        if param_name not in known_params:
            warnings.append(f"Macro '{macro_name}': unknown parameter '{param_name}'")

    return warnings


# ── Main Expander ─────────────────────────────────────────────────────


class MacroExpander:
    """Expands MACRO directives in DSL text.

    Called from dsl_linter.py as Phase 0 of the lint pipeline.
    """

    def __init__(self, registry_path: Path | None = None):
        self._registry = _load_registry(registry_path)
        self._warnings: list[str] = []

    @property
    def warnings(self) -> list[str]:
        """Param validation warnings from the last expand() call."""
        return self._warnings

    def expand(self, text: str) -> tuple[str, list[MacroExpansion]]:
        """Find and expand all MACRO directives in DSL text.

        Returns:
            (expanded_text, list_of_expansions)
        """
        self._warnings = []
        expansions: list[MacroExpansion] = []
        lines = text.split("\n")
        output_lines: list[str] = []
        i = 0

        while i < len(lines):
            m = _MACRO_LINE_RE.match(lines[i])
            if not m or m.group(1).lower() not in self._registry:
                output_lines.append(lines[i])
                i += 1
                continue

            macro_name = m.group(1).lower()
            defn = self._registry[macro_name]
            params = _parse_macro_params(m.group(2))
            self._warnings.extend(validate_macro_params(macro_name, params, defn))
            expanded = _render_template(defn.expansion_template, params, defn)

            if defn.block_macro and defn.end_marker:
                new_i = self._expand_block_macro(
                    lines, i, macro_name, defn, params, expanded, expansions, output_lines
                )
                if new_i is not None:
                    i = new_i
                    continue

            expansions.append(
                MacroExpansion(
                    line=i + 1,
                    macro_name=macro_name,
                    original=lines[i].strip(),
                    expanded=expanded,
                    param_values=params,
                )
            )
            output_lines.extend(expanded.split("\n"))
            i += 1

        return "\n".join(output_lines), expansions

    def _expand_block_macro(
        self,
        lines: list[str],
        start: int,
        macro_name: str,
        defn: MacroDefinition,
        params: dict[str, str],
        expanded: str,
        expansions: list[MacroExpansion],
        output_lines: list[str],
    ) -> int | None:
        """Try to expand a block macro. Returns new line index, or None if end marker not found."""
        j = start + 1
        body_lines = []
        while j < len(lines):
            if lines[j].strip().upper() == defn.end_marker.upper():
                expansion_text = expanded + "\n" + "\n".join(body_lines) + "\n" + defn.end_expansion
                expansions.append(
                    MacroExpansion(
                        line=start + 1,
                        macro_name=macro_name,
                        original="\n".join(lines[start : j + 1]),
                        expanded=expansion_text,
                        param_values=params,
                    )
                )
                output_lines.extend(expansion_text.split("\n"))
                return j + 1
            body_lines.append(lines[j])
            j += 1
        return None


# ── Convenience Functions ─────────────────────────────────────────────


def expand_macros(
    text: str, registry_path: Path | None = None
) -> tuple[str, list[MacroExpansion]]:
    """Convenience function to expand macros in DSL text.

    Returns:
        (expanded_text, list_of_expansions)
    """
    expander = MacroExpander(registry_path)
    return expander.expand(text)


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Expand a DSL file
        path = Path(sys.argv[1])
        text = path.read_text()
        expanded, expansions = expand_macros(text)

        if expansions:
            print(f"Expanded {len(expansions)} macro(s):\n")
            for exp in expansions:
                print(f"  Line {exp.line}: {exp.macro_name}")
                print(f"    Original: {exp.original[:60]}...")
                print(f"    Params: {exp.param_values}")
            print()

        print(expanded)
    else:
        # List available macros
        registry = _load_registry()
        print(f"Available macros ({len(registry)}):\n")
        for name, defn in sorted(registry.items()):
            print(f"  {name}: {defn.description}")
            print(f"    Syntax: {defn.params}")
            if defn.block_macro:
                print(f"    Block macro (end: {defn.end_marker})")
