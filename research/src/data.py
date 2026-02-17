"""
Data loading and dataset classes for Balanced Sashimi training.

Handles JSONL serialization/deserialization through contract types.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from research.src.contracts import NegativeBankEntry, TypedIRExample


def load_typed_ir_jsonl(path: Path) -> list[TypedIRExample]:
    """Load TypedIRExample records from a JSONL file.

    Args:
        path: Path to .jsonl file (one JSON object per line).

    Returns:
        List of TypedIRExample instances.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If a line is malformed.
    """
    examples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                examples.append(TypedIRExample.from_dict(d))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed record at line {line_num}: {e}") from e
    return examples


def save_typed_ir_jsonl(examples: list[TypedIRExample], path: Path) -> int:
    """Save TypedIRExample records to a JSONL file.

    Args:
        examples: Records to save.
        path: Output .jsonl file path.

    Returns:
        Number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return len(examples)


class TypedIRDataset(Dataset):
    """PyTorch Dataset wrapping TypedIRExample records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.examples = load_typed_ir_jsonl(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TypedIRExample:
        return self.examples[idx]


class NegativeBankDataset(Dataset):
    """PyTorch Dataset wrapping NegativeBankEntry records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[NegativeBankEntry] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(NegativeBankEntry.from_dict(json.loads(line)))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> NegativeBankEntry:
        return self.entries[idx]
