#!/usr/bin/env python3
"""
Snippet Extractor: mines reusable 3-12 action sub-sequences from training
data for retrieval-augmented context injection.

Parses training examples, extracts sliding-window sub-sequences, deduplicates
by structural key, scores by frequency/diversity/control-flow, and exports a
snippet registry for use during prompt construction.

Usage:
    python scripts/snippet_extractor.py \
        --input training_data/shortcutdsl_train.jsonl \
        --output references/snippet_registry.json \
        --top-k 200

    python scripts/snippet_extractor.py --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dsl_parser import parse_dsl
from dsl_linter import lint_dsl
from dsl_ir import (
    ShortcutIR,
    ActionStatement,
    SetVariable,
    IfBlock,
    MenuBlock,
    RepeatBlock,
    ForeachBlock,
    Comment,
    VarRef,
    HandleRef,
    InterpolatedString,
    StringValue,
    NumberValue,
    BoolValue,
    DictLiteral,
    ListLiteral,
    QuantityLiteral,
    HeadersLiteral,
    Statement,
    IRValue,
)


# ============================================================
# Domain Tagging
# ============================================================

_ACTION_DOMAIN_MAP: dict[str, str] = {
    # Health
    "health.quantity.log": "health",
    "filter.health.quantity": "health",
    # Networking
    "downloadurl": "networking",
    "url": "networking",
    "geturlcomponent": "networking",
    # Text processing
    "gettext": "text_processing",
    "splittext": "text_processing",
    "replacetext": "text_processing",
    "changecase": "text_processing",
    # Media
    "resizeimage": "media",
    "image.resize": "media",
    "overlayimage": "media",
    "image.overlay": "media",
    "cropimage": "media",
    "image.crop": "media",
    # Messaging
    "sendmessage": "messaging",
    "sendemail": "messaging",
    # Scheduling
    "addnewevent": "scheduling",
    "addnewreminder": "scheduling",
    "getcalendarevents": "scheduling",
    # File operations
    "savefile": "file_operations",
    "documentpicker.save": "file_operations",
    "getfile": "file_operations",
    "documentpicker.open": "file_operations",
    "getfiledetail": "file_operations",
    "file.select": "file_operations",
    # User feedback
    "alert": "user_feedback",
    "showresult": "user_feedback",
    "notification": "user_feedback",
}

# Action name keywords for readable descriptions
_ACTION_READABLE: dict[str, str] = {
    "gettext": "get text",
    "splittext": "split text",
    "replacetext": "replace text",
    "changecase": "change case",
    "downloadurl": "fetch URL",
    "url": "create URL",
    "geturlcomponent": "parse URL component",
    "resizeimage": "resize image",
    "image.resize": "resize image",
    "overlayimage": "overlay image",
    "image.overlay": "overlay image",
    "cropimage": "crop image",
    "image.crop": "crop image",
    "sendmessage": "send message",
    "sendemail": "send email",
    "addnewevent": "add calendar event",
    "addnewreminder": "add reminder",
    "getcalendarevents": "get calendar events",
    "savefile": "save file",
    "documentpicker.save": "save file",
    "getfile": "get file",
    "documentpicker.open": "open file",
    "getfiledetail": "get file detail",
    "alert": "show alert",
    "showresult": "show result",
    "notification": "send notification",
    "ask": "ask for input",
    "getdictionaryvalue": "get dictionary value",
    "detect.dictionary": "parse JSON dictionary",
    "setclipboard": "copy to clipboard",
    "getclipboard": "get clipboard",
    "openurl": "open URL",
    "openapp": "open app",
    "choosefromlist": "choose from list",
    "list": "create list",
    "getitemfromlist": "get item from list",
    "count": "count items",
    "number": "create number",
    "calculate": "calculate",
    "round": "round number",
    "adjustdate": "adjust date",
    "format.date": "format date",
    "getbatterylevel": "get battery level",
    "setvolume": "set volume",
    "setbrightness": "set brightness",
    "share": "share",
    "exit": "exit shortcut",
    "comment": "comment",
    "setvariable": "set variable",
    "getvariable": "get variable",
    "base64encode": "base64 encode",
    "properties.contacts": "get contact property",
    "filter.contacts": "filter contacts",
    "takephoto": "take photo",
    "selectphoto": "select photo",
    "scanbarcode": "scan barcode",
    "speaktext": "speak text",
    "dictation": "dictate text",
    "getrichtextfromhtml": "get rich text from HTML",
    "getmarkdownfromrichtext": "get markdown from rich text",
    "gettimebetweendates": "get time between dates",
    "runworkflow": "run shortcut",
    "getmyworkflows": "get my shortcuts",
}

# Verb-to-action mapping for query matching
_VERB_ACTION_MAP: dict[str, list[str]] = {
    "split": ["splittext"],
    "download": ["downloadurl"],
    "fetch": ["downloadurl"],
    "resize": ["resizeimage", "image.resize"],
    "crop": ["cropimage", "image.crop"],
    "overlay": ["overlayimage", "image.overlay"],
    "email": ["sendemail"],
    "message": ["sendmessage"],
    "text": ["gettext", "splittext", "replacetext"],
    "save": ["savefile", "documentpicker.save"],
    "file": ["savefile", "getfile", "documentpicker.save", "documentpicker.open"],
    "share": ["share"],
    "alert": ["alert"],
    "notify": ["notification"],
    "notification": ["notification"],
    "calendar": ["addnewevent", "getcalendarevents"],
    "reminder": ["addnewreminder"],
    "schedule": ["addnewevent", "addnewreminder"],
    "clipboard": ["setclipboard", "getclipboard"],
    "copy": ["setclipboard"],
    "paste": ["getclipboard"],
    "url": ["url", "openurl", "downloadurl", "geturlcomponent"],
    "open": ["openurl", "openapp"],
    "list": ["list", "choosefromlist", "getitemfromlist"],
    "choose": ["choosefromlist"],
    "photo": ["takephoto", "selectphoto"],
    "image": ["resizeimage", "image.resize", "cropimage", "image.crop"],
    "battery": ["getbatterylevel"],
    "volume": ["setvolume"],
    "brightness": ["setbrightness"],
    "health": ["health.quantity.log", "filter.health.quantity"],
    "json": ["detect.dictionary", "getdictionaryvalue"],
    "dictionary": ["detect.dictionary", "getdictionaryvalue"],
    "parse": ["detect.dictionary", "geturlcomponent"],
    "date": ["adjustdate", "format.date", "gettimebetweendates"],
    "time": ["gettimebetweendates", "adjustdate"],
    "speak": ["speaktext"],
    "dictate": ["dictation"],
    "barcode": ["scanbarcode"],
    "scan": ["scanbarcode"],
    "replace": ["replacetext"],
    "count": ["count"],
    "calculate": ["calculate"],
    "math": ["calculate", "round", "number"],
    "round": ["round"],
    "run": ["runworkflow"],
    "shortcut": ["runworkflow", "getmyworkflows"],
    "loop": [],  # matched by control flow tags
    "repeat": [],
    "iterate": [],
    "condition": [],
    "if": [],
    "menu": [],
}


# ============================================================
# Flat Statement Representation
# ============================================================

@dataclass
class FlatStatement:
    """A statement extracted from IR with its context."""
    kind: str  # "action", "set", "if_start", "else", "endif", "foreach_start",
               # "endforeach", "repeat_start", "endrepeat", "menu_start",
               # "case", "endmenu", "comment"
    action_id: str | None = None  # Only for "action" kind
    params: dict[str, Any] = field(default_factory=dict)
    var_name: str | None = None  # Only for "set" kind
    value_repr: str | None = None  # String representation of the value
    in_control_flow: bool = False  # True if inside IF/FOREACH/REPEAT/MENU
    original_stmt: Statement | None = None


# ============================================================
# IR Flattening
# ============================================================

def _flatten_ir(statements: list[Statement], in_cf: bool = False) -> list[FlatStatement]:
    """Recursively flatten IR statements into a linear list of FlatStatements.

    Walks through control flow blocks, recording their boundaries as
    synthetic flat statements (if_start, endif, foreach_start, etc.)
    so that the sliding window can detect control flow presence.
    """
    result: list[FlatStatement] = []

    for stmt in statements:
        if isinstance(stmt, ActionStatement):
            result.append(FlatStatement(
                kind="action",
                action_id=_normalize_action_id(stmt.action_name),
                params={k: _value_to_str(v) for k, v in stmt.params.items()},
                in_control_flow=in_cf,
                original_stmt=stmt,
            ))
        elif isinstance(stmt, SetVariable):
            result.append(FlatStatement(
                kind="set",
                var_name=stmt.var_name,
                value_repr=_value_to_str(stmt.value),
                in_control_flow=in_cf,
                original_stmt=stmt,
            ))
        elif isinstance(stmt, IfBlock):
            result.append(FlatStatement(kind="if_start", in_control_flow=in_cf))
            result.extend(_flatten_ir(stmt.then_body, in_cf=True))
            if stmt.else_body:
                result.append(FlatStatement(kind="else", in_control_flow=True))
                result.extend(_flatten_ir(stmt.else_body, in_cf=True))
            result.append(FlatStatement(kind="endif", in_control_flow=in_cf))
        elif isinstance(stmt, ForeachBlock):
            result.append(FlatStatement(kind="foreach_start", in_control_flow=in_cf))
            result.extend(_flatten_ir(stmt.body, in_cf=True))
            result.append(FlatStatement(kind="endforeach", in_control_flow=in_cf))
        elif isinstance(stmt, RepeatBlock):
            result.append(FlatStatement(kind="repeat_start", in_control_flow=in_cf))
            result.extend(_flatten_ir(stmt.body, in_cf=True))
            result.append(FlatStatement(kind="endrepeat", in_control_flow=in_cf))
        elif isinstance(stmt, MenuBlock):
            result.append(FlatStatement(kind="menu_start", in_control_flow=in_cf))
            for case in stmt.cases:
                result.append(FlatStatement(kind="case", in_control_flow=True))
                result.extend(_flatten_ir(case.body, in_cf=True))
            result.append(FlatStatement(kind="endmenu", in_control_flow=in_cf))
        elif isinstance(stmt, Comment):
            result.append(FlatStatement(kind="comment", in_control_flow=in_cf))

    return result


def _normalize_action_id(name: str) -> str:
    """Strip the is.workflow.actions. prefix for canonical comparison."""
    prefix = "is.workflow.actions."
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _value_to_str(val: IRValue) -> str:
    """Convert an IR value to a string representation for canonical DSL."""
    if isinstance(val, StringValue):
        return f'"{val.value}"'
    elif isinstance(val, NumberValue):
        return str(val.value)
    elif isinstance(val, BoolValue):
        return "true" if val.value else "false"
    elif isinstance(val, VarRef):
        return f"${val.name}"
    elif isinstance(val, HandleRef):
        return f"@{val.kind}"
    elif isinstance(val, InterpolatedString):
        parts = []
        for p in val.parts:
            if isinstance(p, StringValue):
                parts.append(p.value)
            elif isinstance(p, VarRef):
                parts.append(f"{{${p.name}}}")
            elif isinstance(p, HandleRef):
                parts.append(f"{{@{p.kind}}}")
        return "`" + "".join(parts) + "`"
    elif isinstance(val, DictLiteral):
        entries = ", ".join(f'"{k}": {_value_to_str(v)}' for k, v in val.entries)
        return "{" + entries + "}"
    elif isinstance(val, ListLiteral):
        items = ", ".join(_value_to_str(i) for i in val.items)
        return "[" + items + "]"
    elif isinstance(val, QuantityLiteral):
        if isinstance(val.magnitude, (VarRef, HandleRef)):
            mag = _value_to_str(val.magnitude)
        else:
            mag = str(val.magnitude)
        return f'QTY({mag}, "{val.unit}")'
    elif isinstance(val, HeadersLiteral):
        entries = ", ".join(f'"{k}": {_value_to_str(v)}' for k, v in val.entries)
        return "HEADERS {" + entries + "}"
    return str(val)


# ============================================================
# Variable Canonicalization
# ============================================================

def _canonicalize_variables(flat_stmts: list[FlatStatement]) -> list[FlatStatement]:
    """Replace user-defined variable names with canonical placeholders.

    Variables are renamed in order of first appearance:
    $UserInput -> $__v1, $Response -> $__v2, etc.

    Handles (@prev, @item, etc.) are NOT renamed.
    """
    var_map: dict[str, str] = {}
    counter = 0

    def _remap_var(name: str) -> str:
        nonlocal counter
        if name not in var_map:
            counter += 1
            var_map[name] = f"__v{counter}"
        return var_map[name]

    def _remap_in_str(s: str) -> str:
        """Replace $VarName references in a string."""
        def _replace(m: re.Match) -> str:
            vname = m.group(1)
            return f"${_remap_var(vname)}"
        return re.sub(r'\$([A-Za-z_]\w*)', _replace, s)

    result: list[FlatStatement] = []
    for fs in flat_stmts:
        new_fs = FlatStatement(
            kind=fs.kind,
            action_id=fs.action_id,
            params={k: _remap_in_str(v) for k, v in fs.params.items()},
            var_name=_remap_var(fs.var_name) if fs.var_name else None,
            value_repr=_remap_in_str(fs.value_repr) if fs.value_repr else None,
            in_control_flow=fs.in_control_flow,
            original_stmt=fs.original_stmt,
        )
        result.append(new_fs)
    return result


# ============================================================
# Structural Key & Scoring
# ============================================================

def _compute_structural_key(flat_stmts: list[FlatStatement]) -> str:
    """Compute a dot-joined key from action IDs in the window.

    Only action statements contribute to the key. Control flow markers
    and SET statements are excluded from the key itself but influence
    scoring via has_control_flow detection.
    """
    action_ids = [
        fs.action_id for fs in flat_stmts
        if fs.kind == "action" and fs.action_id
    ]
    return ".".join(action_ids)


_CONTROL_FLOW_KINDS = frozenset({
    "if_start", "else", "endif",
    "foreach_start", "endforeach",
    "repeat_start", "endrepeat",
    "menu_start", "case", "endmenu",
})


def _has_control_flow(flat_stmts: list[FlatStatement]) -> bool:
    """Check if any statement in the window is a control flow marker."""
    return any(fs.kind in _CONTROL_FLOW_KINDS for fs in flat_stmts)


def _score_snippet(frequency: int, action_ids: list[str],
                   has_cf: bool) -> float:
    """Score a snippet candidate.

    score = frequency * action_diversity_ratio * (1 + has_control_flow_weight)

    where:
      - action_diversity_ratio = unique_actions / total_actions
      - has_control_flow_weight = 0.5 if control flow present, else 0
    """
    if not action_ids:
        return 0.0
    total = len(action_ids)
    unique = len(set(action_ids))
    diversity = unique / total
    cf_weight = 0.5 if has_cf else 0.0
    return frequency * diversity * (1.0 + cf_weight)


def _detect_domain_tags(action_ids: list[str]) -> list[str]:
    """Map action IDs to domain tags."""
    tags: set[str] = set()
    for aid in action_ids:
        domain = _ACTION_DOMAIN_MAP.get(aid)
        if domain:
            tags.add(domain)
    if not tags:
        tags.add("general")
    return sorted(tags)


def _generate_description(action_ids: list[str], has_cf: bool) -> str:
    """Auto-generate a human-readable description from action sequence."""
    parts: list[str] = []
    for aid in action_ids:
        readable = _ACTION_READABLE.get(aid, aid.replace(".", " ").replace("_", " "))
        # Capitalize first letter of each part
        readable = readable[0].upper() + readable[1:] if readable else aid
        parts.append(readable)

    desc = ", ".join(parts)
    if has_cf:
        desc += " (with control flow)"
    return desc


# ============================================================
# Canonical DSL Reconstruction
# ============================================================

def _reconstruct_canonical_dsl(flat_stmts: list[FlatStatement]) -> str:
    """Reconstruct a canonical DSL string from flat statements."""
    lines: list[str] = []
    indent = 0

    for fs in flat_stmts:
        prefix = "  " * indent

        if fs.kind == "action":
            param_str = " ".join(f"{k}={v}" for k, v in fs.params.items())
            if param_str:
                lines.append(f"{prefix}ACTION {fs.action_id} {param_str}")
            else:
                lines.append(f"{prefix}ACTION {fs.action_id}")
        elif fs.kind == "set":
            lines.append(f"{prefix}SET ${fs.var_name} = {fs.value_repr}")
        elif fs.kind == "if_start":
            lines.append(f"{prefix}IF ...")
            indent += 1
        elif fs.kind == "else":
            # Dedent for ELSE, then re-indent
            lines.append(f"{'  ' * max(0, indent - 1)}ELSE")
        elif fs.kind == "endif":
            indent = max(0, indent - 1)
            lines.append(f"{'  ' * indent}ENDIF")
        elif fs.kind == "foreach_start":
            lines.append(f"{prefix}FOREACH @prev")
            indent += 1
        elif fs.kind == "endforeach":
            indent = max(0, indent - 1)
            lines.append(f"{'  ' * indent}ENDFOREACH")
        elif fs.kind == "repeat_start":
            lines.append(f"{prefix}REPEAT ...")
            indent += 1
        elif fs.kind == "endrepeat":
            indent = max(0, indent - 1)
            lines.append(f"{'  ' * indent}ENDREPEAT")
        elif fs.kind == "menu_start":
            lines.append(f"{prefix}MENU ...")
            indent += 1
        elif fs.kind == "case":
            lines.append(f"{'  ' * max(0, indent - 1)}CASE ...")
        elif fs.kind == "endmenu":
            indent = max(0, indent - 1)
            lines.append(f"{'  ' * indent}ENDMENU")
        elif fs.kind == "comment":
            pass  # Skip comments in canonical form

    return "\n".join(lines)


# ============================================================
# Snippet Data Structure
# ============================================================

@dataclass
class SnippetCandidate:
    """A candidate snippet before deduplication."""
    structural_key: str
    action_ids: list[str]
    has_control_flow: bool
    canonical_dsl: str
    domain_tags: list[str]
    description: str


# ============================================================
# SnippetExtractor
# ============================================================

WINDOW_SIZES = (3, 5, 7, 10, 12)


class SnippetExtractor:
    """Extract reusable sub-sequences from training data.

    Pipeline:
      1. Parse each training example -> IR
      2. Flatten IR to flat statements
      3. Sliding window extraction
      4. Deduplication by structural key
      5. Scoring and top-K selection
      6. Export to snippet registry JSON
    """

    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self._key_counts: Counter[str] = Counter()
        self._key_candidates: dict[str, SnippetCandidate] = {}
        self._parse_errors = 0
        self._total_examples = 0
        self._total_windows = 0

    def ingest_dsl(self, dsl_text: str) -> None:
        """Parse a single DSL text and extract snippet candidates."""
        self._total_examples += 1

        try:
            # Lint before parsing
            lint_result = lint_dsl(dsl_text)
            ir = parse_dsl(lint_result.text)
        except Exception:
            self._parse_errors += 1
            return

        flat = _flatten_ir(ir.statements)

        if len(flat) < 3:
            return

        for wsize in WINDOW_SIZES:
            if wsize > len(flat):
                continue
            for start in range(len(flat) - wsize + 1):
                window = flat[start:start + wsize]
                self._process_window(window)

    def _process_window(self, window: list[FlatStatement]) -> None:
        """Process a single sliding window of flat statements."""
        self._total_windows += 1

        # Extract action IDs (only from action statements)
        action_ids = [
            fs.action_id for fs in window
            if fs.kind == "action" and fs.action_id
        ]

        # Skip windows with fewer than 2 unique action IDs
        if len(set(action_ids)) < 2:
            return

        # Canonicalize variables
        canonical_window = _canonicalize_variables(window)

        # Compute structural key
        key = _compute_structural_key(canonical_window)
        if not key:
            return

        # Record occurrence
        self._key_counts[key] += 1

        # Store first candidate for each key (representative)
        if key not in self._key_candidates:
            has_cf = _has_control_flow(canonical_window)
            self._key_candidates[key] = SnippetCandidate(
                structural_key=key,
                action_ids=action_ids,
                has_control_flow=has_cf,
                canonical_dsl=_reconstruct_canonical_dsl(canonical_window),
                domain_tags=_detect_domain_tags(action_ids),
                description=_generate_description(action_ids, has_cf),
            )

    def extract(self) -> list[dict]:
        """Score, rank, and return top-K snippets as dictionaries."""
        scored: list[tuple[float, str]] = []

        for key, candidate in self._key_candidates.items():
            freq = self._key_counts[key]
            score = _score_snippet(
                freq, candidate.action_ids, candidate.has_control_flow
            )
            scored.append((score, key))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Take top-K
        top = scored[:self.top_k]

        snippets: list[dict] = []
        for rank, (score, key) in enumerate(top, 1):
            candidate = self._key_candidates[key]
            snippets.append({
                "id": f"snip_{rank:03d}",
                "structural_key": key,
                "canonical_dsl": candidate.canonical_dsl,
                "action_count": len(candidate.action_ids),
                "frequency": self._key_counts[key],
                "score": round(score, 4),
                "domain_tags": candidate.domain_tags,
                "description": candidate.description,
            })

        return snippets

    def build_registry(self, source_path: str) -> dict:
        """Build the full registry dictionary."""
        snippets = self.extract()
        return {
            "version": "1.0",
            "extracted_from": source_path,
            "snippet_count": len(snippets),
            "snippets": snippets,
        }

    def stats(self) -> dict:
        """Return extraction statistics."""
        return {
            "total_examples": self._total_examples,
            "parse_errors": self._parse_errors,
            "parsed_ok": self._total_examples - self._parse_errors,
            "total_windows_evaluated": self._total_windows,
            "unique_structural_keys": len(self._key_candidates),
            "top_k": self.top_k,
        }


# ============================================================
# Query Function
# ============================================================

def query_snippets(
    prompt: str,
    registry_path: Path | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Query the snippet registry for relevant snippets.

    Matches prompt keywords to action IDs via a verb mapping, then
    scores snippet matches by keyword overlap * quality score.

    Args:
        prompt: Natural language prompt describing the desired shortcut.
        registry_path: Path to snippet_registry.json. Defaults to
            references/snippet_registry.json.
        top_k: Number of results to return.

    Returns:
        List of snippet dicts sorted by relevance score (highest first).
    """
    if registry_path is None:
        registry_path = (
            Path(__file__).resolve().parent.parent
            / "references"
            / "snippet_registry.json"
        )

    if not registry_path.exists():
        return []

    with open(registry_path) as f:
        registry = json.load(f)

    snippets = registry.get("snippets", [])
    if not snippets:
        return []

    # Tokenize prompt into lowercase words
    prompt_words = set(re.findall(r'[a-z]+', prompt.lower()))

    # Map prompt words to action IDs via verb mapping
    target_actions: set[str] = set()
    matched_domains: set[str] = set()
    has_control_flow_keyword = False

    for word in prompt_words:
        if word in _VERB_ACTION_MAP:
            target_actions.update(_VERB_ACTION_MAP[word])
        # Check for control flow keywords
        if word in {"loop", "repeat", "iterate", "condition", "if", "menu",
                     "foreach", "each", "every"}:
            has_control_flow_keyword = True
        # Direct domain matching
        for domain_name in ("health", "networking", "text_processing", "media",
                            "messaging", "scheduling", "file_operations",
                            "user_feedback"):
            # Match on domain name words
            for dw in domain_name.split("_"):
                if dw in prompt_words:
                    matched_domains.add(domain_name)

    if not target_actions and not matched_domains and not has_control_flow_keyword:
        # No recognized keywords -- return top snippets by score
        sorted_by_score = sorted(snippets, key=lambda s: -s.get("score", 0))
        return sorted_by_score[:top_k]

    # Score each snippet by relevance
    scored: list[tuple[float, dict]] = []

    for snip in snippets:
        key_actions = set(snip.get("structural_key", "").split("."))
        snip_domains = set(snip.get("domain_tags", []))

        # Keyword overlap: how many target actions appear in the snippet key
        if target_actions:
            action_overlap = len(target_actions & key_actions) / len(target_actions)
        else:
            action_overlap = 0.0

        # Domain overlap
        if matched_domains:
            domain_overlap = len(matched_domains & snip_domains) / len(matched_domains)
        else:
            domain_overlap = 0.0

        # Control flow bonus
        cf_bonus = 0.0
        if has_control_flow_keyword:
            canonical = snip.get("canonical_dsl", "")
            if any(kw in canonical for kw in ("IF", "FOREACH", "REPEAT", "MENU")):
                cf_bonus = 0.3

        # Combined relevance = keyword overlap * snippet quality score
        raw_relevance = max(action_overlap, domain_overlap) + cf_bonus
        quality = snip.get("score", 0.0)
        relevance = raw_relevance * (1.0 + quality * 0.1)

        if relevance > 0:
            scored.append((relevance, snip))

    # Sort by relevance descending
    scored.sort(key=lambda x: -x[0])

    return [s[1] for s in scored[:top_k]]


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract reusable snippet sub-sequences from training data."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "training_data" / "shortcutdsl_train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "references" / "snippet_registry.json",
        help="Path to output snippet registry JSON",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=200,
        help="Number of top snippets to export (default: 200)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print extraction stats only (don't write output)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    extractor = SnippetExtractor(top_k=args.top_k)

    # Load and process training examples
    print(f"Loading training data from {args.input}...")
    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: invalid JSON on line {line_num}", file=sys.stderr)
                continue

            # Extract DSL from the assistant message
            messages = example.get("messages", [])
            dsl_text = None
            for msg in messages:
                if msg.get("role") == "assistant":
                    dsl_text = msg.get("content", "")
                    break

            if not dsl_text:
                continue

            extractor.ingest_dsl(dsl_text)

            if line_num % 100 == 0:
                print(f"  Processed {line_num} examples...")

    # Print stats
    stats = extractor.stats()
    print(f"\nExtraction Statistics:")
    print(f"  Total examples:          {stats['total_examples']}")
    print(f"  Parse errors:            {stats['parse_errors']}")
    print(f"  Successfully parsed:     {stats['parsed_ok']}")
    print(f"  Windows evaluated:       {stats['total_windows_evaluated']}")
    print(f"  Unique structural keys:  {stats['unique_structural_keys']}")

    if args.stats:
        return

    # Build and write registry
    registry = extractor.build_registry(str(args.input))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nExported {registry['snippet_count']} snippets to {args.output}")


if __name__ == "__main__":
    main()
