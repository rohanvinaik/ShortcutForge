"""Cross-tokenizer normalization for action prediction comparison.

Maps raw model predictions to canonical action names from the ShortcutForge
action catalog, enabling cross-model behavioral comparison despite different
tokenizers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_ACTION_LINE_RE = re.compile(r"ACTION\s+([\w.]+)", re.IGNORECASE)


class ActionVocab:
    """Canonical action vocabulary loaded from the ShortcutForge action catalog."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        if catalog_path is None:
            catalog_path = PROJECT_ROOT / "references" / "action_catalog.json"

        with open(catalog_path) as f:
            catalog = json.load(f)

        # Canonical action identifiers
        actions_section = catalog.get("actions", {})
        self.actions: set[str] = set(actions_section.keys())

        # Alias map: short name -> canonical identifier
        meta = catalog.get("_meta", {})
        self.aliases: dict[str, str] = dict(meta.get("canonical_map", {}))

    def resolve(self, name: str) -> str | None:
        """Resolve a name to its canonical action identifier.

        Tries exact match first, then alias lookup (case-insensitive).
        Returns None if no match is found.
        """
        if name in self.actions:
            return name
        lower = name.lower()
        if lower in self.aliases:
            canonical = self.aliases[lower]
            if canonical in self.actions:
                return canonical
        return None


def extract_first_action(generated_text: str) -> str | None:
    """Extract the first ACTION identifier from generated DSL text.

    Looks for lines matching ``ACTION <identifier>`` and returns the
    identifier. Returns None if no action line is found.
    """
    match = _ACTION_LINE_RE.search(generated_text)
    if match:
        return match.group(1)
    return None


def normalize_predictions(predictions: list[str], action_vocab: ActionVocab) -> list[str]:
    """Normalize raw model outputs to canonical action names.

    For each generated text, extracts the first action identifier and
    resolves it through the action vocabulary. Uses ``<UNKNOWN>`` for
    predictions that cannot be resolved.
    """
    results = []
    for text in predictions:
        action_id = extract_first_action(text)
        if action_id is None:
            results.append("<UNKNOWN>")
            continue
        canonical = action_vocab.resolve(action_id)
        results.append(canonical if canonical is not None else "<UNKNOWN>")
    return results
