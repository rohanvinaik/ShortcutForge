#!/usr/bin/env python3
"""
DSL Linter — comprehensive repair layer for LLM-generated ShortcutDSL.

Operates on raw DSL text BEFORE the Lark parser, fixing all classes of
LLM generation errors that can be deterministically repaired:

  TEXT-LEVEL (operate on raw string before any line parsing):
    1. Multi-line interpolation collapse — newlines inside `...` → \\n
    2. Unbalanced backtick repair — close or remove orphan backticks
    3. Markdown fence stripping — remove ```dsl ... ``` wrappers
    4. Preamble/postamble cleanup — strip chat around SHORTCUT...

  LINE-LEVEL (operate on each line):
    5. Hallucinated action names → nearest valid action (fuzzy match)
    6. Invalid condition keywords → nearest valid condition (alias + fuzzy)
    7. Handle/var property access stripping — @prev.Name → @prev
    8. Incomplete ACTION lines — ACTION without name → remove
    9. Bare PARAM lines — PARAM without parent ACTION → rewrite as ACTION params

  STRUCTURAL (operate on block nesting):
    10. Unclosed blocks from truncation → auto-close IF/MENU/FOREACH/REPEAT
    11. Orphan ELSE/CASE outside blocks → remove

  FINAL:
    12. Trailing newline — grammar requires it

Usage:
    from dsl_linter import lint_dsl
    result = lint_dsl(raw_text)
    # result.text    — repaired DSL
    # result.changes — list of LintChange records
    # result.was_modified — bool
"""

from __future__ import annotations

__version__ = "2.4"

import json
import re
from dataclasses import dataclass, field
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path
from typing import Optional

from macro_expander import MacroExpander, MacroExpansion


# ── Constants ────────────────────────────────────────────────────────

VALID_CONDITIONS = frozenset({
    "has_any_value",
    "does_not_have_any_value",
    "equals_number",
    "is_greater_than",
    "is_less_than",
    "equals_string",
    "not_equal_string",
    "contains",
    "does_not_contain",
    "is_before",
})

# Common hallucination → correct condition mappings
CONDITION_ALIASES = {
    # Semantic equivalents
    "has_any_tag": "has_any_value",
    "has_value": "has_any_value",
    "is_not_empty": "has_any_value",
    "has_no_value": "does_not_have_any_value",
    "is_empty": "does_not_have_any_value",
    "does_not_have_value": "does_not_have_any_value",
    "no_value": "does_not_have_any_value",
    # Numeric comparisons
    "is_larger_than": "is_greater_than",
    "is_bigger_than": "is_greater_than",
    "is_more_than": "is_greater_than",
    "greater_than": "is_greater_than",
    "is_above": "is_greater_than",
    "above": "is_greater_than",
    "is_smaller_than": "is_less_than",
    "less_than": "is_less_than",
    "is_below": "is_less_than",
    "below": "is_less_than",
    "is_fewer_than": "is_less_than",
    # Equality
    "equals": "equals_string",
    "equal": "equals_string",
    "is_equal_to": "equals_string",
    "is": "equals_string",
    "matches": "equals_string",
    "equal_to": "equals_string",
    "not_equal": "not_equal_string",
    "is_not_equal_to": "not_equal_string",
    "not_equals": "not_equal_string",
    "is_not": "not_equal_string",
    "does_not_equal": "not_equal_string",
    "not_equal_to": "not_equal_string",
    "doesnt_equal": "not_equal_string",
    # Containment
    "not_contains": "does_not_contain",
    "doesnt_contain": "does_not_contain",
    "includes": "contains",
    "has": "contains",
    "does_not_include": "does_not_contain",
    # Date
    "is_after": "is_before",   # semantically different, but grammatically valid
    "before": "is_before",
    "after": "is_before",      # same note as above
    # Numeric equality (map to equals_number)
    "equals_num": "equals_number",
    "is_equal_number": "equals_number",
    "equal_number": "equals_number",
}

# DSL structural keywords — used to detect line boundaries
_STRUCTURAL_KEYWORDS = frozenset({
    "SHORTCUT", "ACTION", "SET", "IF", "ELSE", "ENDIF",
    "MENU", "CASE", "ENDMENU", "REPEAT", "ENDREPEAT",
    "FOREACH", "ENDFOREACH", "ENDSHORTCUT",
})


@dataclass
class LintChange:
    """A single repair made by the linter."""
    line: int
    kind: str  # "action", "alias_warning", "condition", "structure", "interpolation", "handle", "trailing_newline"
    original: str
    replacement: str
    confidence: float  # 0.0 to 1.0
    reason: str = ""  # Human-readable explanation of why this change was made


@dataclass
class LintResult:
    """Result of linting: fixed text + list of changes made."""
    text: str
    changes: list[LintChange] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return len(self.changes) > 0


# ── Action Name Resolution ──────────────────────────────────────────

class ActionResolver:
    """Resolves action names using the validator's catalog + fuzzy matching.

    Resolution order (first match wins):
      1. Exact match in catalog actions
      2. Exact match in canonical_map aliases
      3. Auto-prefix: is.workflow.actions.{name} in catalog
      4. Namespace-aware fuzzy match (segment-by-segment similarity)
      5. Global fuzzy match (SequenceMatcher ratio)
    """

    # Common model hallucinations → correct action names.
    # These are semantic equivalents that differ too much textually
    # for fuzzy matching to find them reliably.
    #
    # Organized by category. ~80+ entries covering all observed failure patterns.
    HALLUCINATION_ALIASES = {
        # ── URL/Web Actions ────────────────────────────────────────
        "is.workflow.actions.getcontentofurl": "is.workflow.actions.downloadurl",
        "is.workflow.actions.getcontentsofurl": "is.workflow.actions.downloadurl",
        "is.workflow.actions.urlcontents": "is.workflow.actions.downloadurl",
        "is.workflow.actions.fetchurl": "is.workflow.actions.downloadurl",
        "is.workflow.actions.httpget": "is.workflow.actions.downloadurl",
        "is.workflow.actions.webrequest": "is.workflow.actions.downloadurl",
        "getcontentofurl": "downloadurl",
        "getcontentsofurl": "downloadurl",
        "fetchurl": "downloadurl",
        "httpget": "downloadurl",
        "webrequest": "downloadurl",
        "urlcontents": "downloadurl",
        # ── Battery ────────────────────────────────────────────────
        "is.workflow.actions.getbatterystate": "is.workflow.actions.getbatterylevel",
        "getbatterystate": "getbatterylevel",
        "is.workflow.actions.batterystate": "is.workflow.actions.getbatterylevel",
        "batterystate": "getbatterylevel",
        "batterylevel": "getbatterylevel",
        "checkbattery": "getbatterylevel",
        # ── Low Power Mode ─────────────────────────────────────────
        "is.workflow.actions.setLowPowerMode": "is.workflow.actions.lowpowermode.set",
        "is.workflow.actions.setlowpowermode": "is.workflow.actions.lowpowermode.set",
        "is.workflow.actions.lowpowermode": "is.workflow.actions.lowpowermode.set",
        "setlowpowermode": "lowpowermode.set",
        "enablelowpowermode": "lowpowermode.set",
        "lowpowermode": "lowpowermode.set",
        # ── Clipboard ──────────────────────────────────────────────
        "is.workflow.actions.getclipboardcontents": "is.workflow.actions.getclipboard",
        "is.workflow.actions.setclipboardcontents": "is.workflow.actions.setclipboard",
        "is.workflow.actions.copytoclipboard": "is.workflow.actions.setclipboard",
        "getclipboardcontents": "getclipboard",
        "setclipboardcontents": "setclipboard",
        "copytoclipboard": "setclipboard",
        "clipboard": "getclipboard",
        "pasteboard": "getclipboard",
        # ── Notifications ──────────────────────────────────────────
        "is.workflow.actions.sendnotification": "is.workflow.actions.notification",
        "is.workflow.actions.shownotification": "is.workflow.actions.notification",
        "sendnotification": "notification",
        "shownotification": "notification",
        "pushnotification": "notification",
        "localnotification": "notification",
        # ── WiFi / Bluetooth / Airplane / Cellular ──────────────────
        "is.workflow.actions.setwifi": "is.workflow.actions.wifi.set",
        "is.workflow.actions.setbluetooth": "is.workflow.actions.bluetooth.set",
        "is.workflow.actions.setairplanemode": "is.workflow.actions.airplanemode.set",
        "is.workflow.actions.setcellulardata": "is.workflow.actions.cellulardata.set",
        "is.workflow.actions.togglewifi": "is.workflow.actions.wifi.set",
        "is.workflow.actions.togglebluetooth": "is.workflow.actions.bluetooth.set",
        "is.workflow.actions.toggleairplanemode": "is.workflow.actions.airplanemode.set",
        "is.workflow.actions.togglecellulardata": "is.workflow.actions.cellulardata.set",
        "setwifi": "wifi.set",
        "setbluetooth": "bluetooth.set",
        "setairplanemode": "airplanemode.set",
        "setcellulardata": "cellulardata.set",
        "togglewifi": "wifi.set",
        "togglebluetooth": "bluetooth.set",
        "toggleairplanemode": "airplanemode.set",
        "togglecellulardata": "cellulardata.set",
        "turnwifion": "wifi.set",
        "turnwifioff": "wifi.set",
        "turnbluetoothon": "bluetooth.set",
        "turnbluetoothoff": "bluetooth.set",
        "enablewifi": "wifi.set",
        "disablewifi": "wifi.set",
        "enablebluetooth": "bluetooth.set",
        "disablebluetooth": "bluetooth.set",
        # ── Flashlight ─────────────────────────────────────────────
        "flashon": "flashlight",
        "flashoff": "flashlight",
        "toggleflashlight": "flashlight",
        "setflashlight": "flashlight",
        "flashlight.set": "flashlight",
        "turnflashlighton": "flashlight",
        "turnflashlightoff": "flashlight",
        "enableflashlight": "flashlight",
        "is.workflow.actions.flashon": "is.workflow.actions.flashlight",
        "is.workflow.actions.flashoff": "is.workflow.actions.flashlight",
        "is.workflow.actions.toggleflashlight": "is.workflow.actions.flashlight",
        "is.workflow.actions.setflashlight": "is.workflow.actions.flashlight",
        # ── Timer / Clock / Stopwatch / Alarm ──────────────────────
        "is.workflow.actions.settimer": "is.workflow.actions.timer.start",
        "settimer": "timer.start",
        "starttimer": "timer.start",
        "newtimer": "timer.start",
        "createtimer": "timer.start",
        "is.workflow.actions.starttimer": "is.workflow.actions.timer.start",
        # Stopwatch (model hallucinates short names; real action is system intent)
        "startstopwatch": "startstopwatch",   # maps to CM → com.apple.mobiletimer.StartStopwatchIntent
        "stopstopwatch": "stopstopwatch",     # maps to CM
        "stopwatch": "startstopwatch",        # generic → start
        "launchstopwatch": "startstopwatch",
        # Alarm (model hallucinates short names; real action is system intent)
        "alarm": "createalarm",               # maps to CM → MTCreateAlarmIntent
        "setalarm": "createalarm",
        "newalarm": "createalarm",
        "addalarm": "createalarm",
        "makealarm": "createalarm",
        "is.workflow.actions.alarm": "is.workflow.actions.timer.start",
        "is.workflow.actions.setalarm": "is.workflow.actions.timer.start",
        # ── Volume ─────────────────────────────────────────────────
        "volume": "setvolume",
        "changevolume": "setvolume",
        "adjustvolume": "setvolume",
        # ── Brightness ─────────────────────────────────────────────
        "is.workflow.actions.setscreenbrightness": "is.workflow.actions.setbrightness",
        "setscreenbrightness": "setbrightness",
        "screenbrightness": "setbrightness",
        "changebrightness": "setbrightness",
        "brightness": "setbrightness",
        # ── Maps ───────────────────────────────────────────────────
        "is.workflow.actions.showinmaps": "is.workflow.actions.searchmaps",
        "showinmaps": "searchmaps",
        "is.workflow.actions.showonmap": "is.workflow.actions.searchmaps",
        "showonmap": "searchmaps",
        "openmaps": "searchmaps",
        "findlocation": "searchmaps",
        # ── Camera / Photo / Media ─────────────────────────────────
        "takeselfie": "takephoto",
        "selfie": "takephoto",
        "captureimage": "takephoto",
        "capturephoto": "takephoto",
        "is.workflow.actions.takeselfie": "is.workflow.actions.takephoto",
        "recordvideo": "takevideo",
        "capturevideo": "takevideo",
        "is.workflow.actions.recordvideo": "is.workflow.actions.takevideo",
        "is.workflow.actions.capturevideo": "is.workflow.actions.takevideo",
        "screenshot": "takescreenshot",
        "getscreenshot": "takescreenshot",
        "capturescreenshot": "takescreenshot",
        "capturescreen": "takescreenshot",
        "is.workflow.actions.screenshot": "is.workflow.actions.takescreenshot",
        # Voice memo
        "recordvoicememo": "recordvoicememo",  # maps to CM → VoiceMemos.RecordVoiceMemoIntent
        "voicememo": "recordvoicememo",
        "startrecording": "recordvoicememo",
        # ── Image Operations ───────────────────────────────────────
        "crop": "image.crop",
        "is.workflow.actions.crop": "is.workflow.actions.image.crop",
        "resize": "image.resize",
        "is.workflow.actions.resize": "is.workflow.actions.image.resize",
        "rotateimage": "image.rotate",
        "flipimage": "image.flip",
        # Media selection (wrong name)
        "selectmedia": "selectphoto",
        "is.workflow.actions.selectmedia": "is.workflow.actions.selectphoto",
        "choosephoto": "selectphoto",
        "pickphoto": "selectphoto",
        # ── Reminders / Calendar ───────────────────────────────────
        "reminders.add": "addnewreminder",
        "is.workflow.actions.reminders.add": "is.workflow.actions.addnewreminder",
        "createreminder": "addnewreminder",
        "addreminder": "addnewreminder",
        "newreminder": "addnewreminder",
        "addevent": "addnewevent",
        "createevent": "addnewevent",
        "newevent": "addnewevent",
        "addcalendarevent": "addnewevent",
        # ── Notes / Text ───────────────────────────────────────────
        "createnote": "appendnote",
        "newnote": "appendnote",
        "addnote": "appendnote",
        "savenote": "appendnote",
        # ── Bookmarks / URL ────────────────────────────────────────
        "is.workflow.actions.openbookmark": "is.workflow.actions.openurl",
        "openbookmark": "openurl",
        "openlink": "openurl",
        "openwebpage": "openurl",
        # ── Rich text / Markdown ───────────────────────────────────
        "is.workflow.actions.getrichtextfrommarkdown": "is.workflow.actions.getmarkdownfromrichtext",
        "getrichtextfrommarkdown": "getmarkdownfromrichtext",
        # ── Time calculations ──────────────────────────────────────
        "gettimeuntil": "gettimebetweendates",
        "is.workflow.actions.gettimeuntil": "is.workflow.actions.gettimebetweendates",
        "timebetween": "gettimebetweendates",
        "timedifference": "gettimebetweendates",
        # ── Do Not Disturb / Focus ─────────────────────────────────
        "donotdisturb": "dnd.set",
        "togglednd": "dnd.set",
        "setdnd": "dnd.set",
        "focusmode": "dnd.set",
        "setfocus": "dnd.set",
        "togglefocus": "dnd.set",
        "is.workflow.actions.donotdisturb": "is.workflow.actions.dnd.set",
        "is.workflow.actions.setdnd": "is.workflow.actions.dnd.set",
        # ── Appearance ─────────────────────────────────────────────
        "darkmode": "appearance",
        "lightmode": "appearance",
        "setdarkmode": "appearance",
        "setlightmode": "appearance",
        "toggledarkmode": "appearance",
        "setappearance": "appearance",
        # ── DSL Keyword Confusion (model emits keywords as action names) ──
        "menu": "choosefrommenu",
        "repeat_forever": "choosefrommenu",  # model confuses repeat_forever; no real equivalent
        # ── Text / Input ───────────────────────────────────────────
        "getinput": "ask",
        "askforinput": "ask",
        "promptuser": "ask",
        "userinput": "ask",
        "textinput": "ask",
        # ── Speak / Speech ─────────────────────────────────────────
        "speaktext": "speaktext",
        "speak": "speaktext",
        "saytext": "speaktext",
        "texttospeech": "speaktext",
        "tts": "speaktext",
        "readaloud": "speaktext",
        # ── AirDrop / Share ────────────────────────────────────────
        "airdrop": "airdropdocument",
        "sendairdrop": "airdropdocument",
        "shareairdrop": "airdropdocument",
        # ── Contacts ───────────────────────────────────────────────
        "addcontact": "addnewcontact",
        "createcontact": "addnewcontact",
        "newcontact": "addnewcontact",
        # ── AirBuddy (vendor name correction) ──────────────────────
        "com.sindreselass.AirBuddy.Mobile.AirBuddyIntent": "codes.rambo.AirBuddyHelper.ConnectHeadsetIntent",
        "com.sindreselass.AirBuddy.StartListeningIntent": "codes.rambo.AirBuddyHelper.ConnectHeadsetIntent",
        # ── Corsair (truncated intent name) ────────────────────────
        "com.corsair.ios.ControlCenter.Widget": "com.corsair.ios.ControlCenter.WidgetDeviceActionIntent",
        # ── App Operations ─────────────────────────────────────────
        "launchapp": "openapp",
        "startapp": "openapp",
        "runapp": "openapp",
        # ── Files / Documents ──────────────────────────────────────
        "savefile": "documentpicker.save",
        "openfile": "documentpicker.open",
        "selectfile": "file.select",
        "choosefile": "file.select",
        "pickfile": "file.select",
        # ── Messaging ──────────────────────────────────────────────
        "sendtext": "sendmessage",
        "textmessage": "sendmessage",
        "sms": "sendmessage",
        "imessage": "sendmessage",
        # ── Music / Playback ───────────────────────────────────────
        "playmusic": "pausemusic",       # pausemusic = play/pause toggle
        "playpause": "pausemusic",
        "toggleplayback": "pausemusic",
        "playsong": "pausemusic",
        "resumemusic": "pausemusic",
        "currentsong": "getcurrentsong",
        "nowplaying": "getcurrentsong",
        # ── Wallet / Payment ───────────────────────────────────────
        "showwallet": "openapp",  # no wallet-specific action; open Wallet app
        # ── QR / Barcode ───────────────────────────────────────────
        "scanqr": "scanbarcode",
        "scanqrcode": "scanbarcode",
        "readqr": "scanbarcode",
        "readbarcode": "scanbarcode",
    }

    # Semantic-risky aliases: rewriting changes the action's semantics.
    # These are mapped with lower confidence and an explanatory reason.
    # Format: { hallucinated_name: (correct_name, reason_string) }
    SEMANTIC_RISKY_ALIASES = {
        "convertlivephoto": (
            "getlatestlivephotos",
            "implies conversion but maps to a fetch action; verify intent",
        ),
        "is.workflow.actions.convertlivephoto": (
            "is.workflow.actions.getlatestlivephotos",
            "implies conversion but maps to a fetch action; verify intent",
        ),
    }

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = str(
                Path(__file__).resolve().parent.parent
                / "references"
                / "action_catalog.json"
            )

        with open(catalog_path) as f:
            catalog = json.load(f)

        self._actions = set(catalog.get("actions", {}).keys())
        self._canonical_map = catalog.get("_meta", {}).get("canonical_map", {})

        # Merge hallucination aliases into canonical map
        self._canonical_map.update(self.HALLUCINATION_ALIASES)

        # Build the full set of resolvable names
        self._all_names: set[str] = set()
        for a in self._actions:
            self._all_names.add(a)
            if a.startswith("is.workflow.actions."):
                self._all_names.add(a[len("is.workflow.actions."):])
        for k in self._canonical_map:
            self._all_names.add(k)

        # Pre-split into short names and long names for faster matching
        self._short_names = sorted(n for n in self._all_names if n.count(".") <= 2)
        self._long_names = sorted(n for n in self._all_names if n.count(".") > 2)

        # Build namespace index for segment-aware matching
        # e.g., "com.apple" → [all names starting with com.apple.*]
        self._namespace_index: dict[str, list[str]] = {}
        for name in self._long_names:
            parts = name.split(".")
            for depth in range(2, len(parts)):
                ns = ".".join(parts[:depth])
                self._namespace_index.setdefault(ns, []).append(name)

    def is_valid(self, name: str) -> bool:
        """Check if an action name is directly valid (no alias resolution needed)."""
        if name in self._actions:
            return True
        if f"is.workflow.actions.{name}" in self._actions:
            return True
        return False

    def is_resolvable(self, name: str) -> bool:
        """Check if an action name can be resolved (valid OR has alias)."""
        if self.is_valid(name):
            return True
        if name in self._canonical_map:
            return True
        if name in self.SEMANTIC_RISKY_ALIASES:
            return True
        return False

    def is_semantic_risky(self, name: str) -> tuple[bool, str]:
        """Check if an action name is a semantic-risky alias.

        Returns (is_risky, reason). If is_risky is True, the name maps to a
        valid action but with a semantic mismatch that the user should verify.
        """
        if name in self.SEMANTIC_RISKY_ALIASES:
            _, reason = self.SEMANTIC_RISKY_ALIASES[name]
            return True, reason
        return False, ""

    def find_closest(self, name: str, cutoff: float = 0.6) -> tuple[Optional[str], bool, str]:
        """Find the closest valid action name using multi-strategy matching.

        Strategies (in order):
          0. Exact match in catalog or canonical_map (incl. hallucination aliases)
          0b. Semantic-risky alias lookup (lower confidence)
          1. Namespace-scoped fuzzy: if name has dots, find candidates sharing
             the longest common namespace prefix, then fuzzy within that scope.
          2. Global fuzzy: search all names at the appropriate length tier.
          3. Cross-tier fallback: if no match in primary tier, try the other.

        Returns (best_match, is_alias, reason):
          - is_alias=True means the match came from a trusted canonical mapping
            (not fuzzy), so suffix validation can be skipped.
          - reason is non-empty for semantic-risky aliases and fuzzy matches.
        """
        if name in self._actions:
            return name, False, ""
        # Strategy 0: Canonical/alias lookup (returns the *target*, not the alias)
        if name in self._canonical_map:
            return self._canonical_map[name], True, f"hallucination alias: {name} is a known synonym for {self._canonical_map[name]}"
        if f"is.workflow.actions.{name}" in self._actions:
            return name, False, ""

        # Strategy 0b: Semantic-risky alias lookup
        if name in self.SEMANTIC_RISKY_ALIASES:
            target, reason = self.SEMANTIC_RISKY_ALIASES[name]
            return target, True, f"semantic-risky alias: {name} → {target}; {reason}"

        # Strategy 1: Namespace-scoped fuzzy for long names
        if name.count(".") > 2:
            best = self._namespace_fuzzy(name, cutoff)
            if best:
                suffix_sim = _suffix_similarity(name, best)
                return best, False, f"fuzzy match ({_similarity(name, best):.2f} similarity, suffix-sim {suffix_sim:.2f})"

        # Strategy 2: Global fuzzy in the matching tier
        if name.count(".") > 2:
            candidates = self._long_names
        else:
            candidates = self._short_names

        matches = get_close_matches(name, candidates, n=1, cutoff=cutoff)
        if matches:
            suffix_sim = _suffix_similarity(name, matches[0])
            return matches[0], False, f"fuzzy match ({_similarity(name, matches[0]):.2f} similarity, suffix-sim {suffix_sim:.2f})"

        # Strategy 3: Cross-tier fallback
        fallback = self._long_names if candidates is self._short_names else self._short_names
        matches = get_close_matches(name, fallback, n=1, cutoff=cutoff)
        if matches:
            suffix_sim = _suffix_similarity(name, matches[0])
            return matches[0], False, f"fuzzy match ({_similarity(name, matches[0]):.2f} similarity, suffix-sim {suffix_sim:.2f})"

        return None, False, ""

    def _namespace_fuzzy(self, name: str, cutoff: float) -> Optional[str]:
        """Namespace-aware fuzzy match for dotted action names.

        Finds the longest namespace prefix shared with catalog entries,
        then fuzzy-matches the remaining suffix within that scope.
        This handles cases like:
          com.sindreselass.AirBuddy → com.sindresorhus.AirBuddy
          com.omnigroup.OmniFocusforMac.Entity → com.omnigroup.OmniFocus3.MacAppStore.AddTaskIntent
        """
        parts = name.split(".")
        # Try progressively shorter prefixes (longest common namespace first)
        for depth in range(min(len(parts) - 1, 4), 0, -1):
            ns = ".".join(parts[:depth])
            candidates = self._namespace_index.get(ns, [])
            if candidates:
                matches = get_close_matches(name, candidates, n=1, cutoff=cutoff)
                if matches:
                    return matches[0]

        # Also try matching with just the first segment (vendor match)
        if len(parts) >= 2:
            vendor = parts[0] + "." + parts[1]
            candidates = self._namespace_index.get(vendor, [])
            if not candidates and parts[0] == "com":
                # Try broader vendor search: all com.X.* where X is fuzzy-close
                for ns_key, ns_candidates in self._namespace_index.items():
                    if ns_key.startswith("com.") and ns_key.count(".") == 1:
                        if _similarity(vendor, ns_key) > 0.7:
                            matches = get_close_matches(name, ns_candidates, n=1, cutoff=cutoff - 0.05)
                            if matches:
                                return matches[0]

        return None


# ── Linting Functions ────────────────────────────────────────────────

# Regex patterns
_ACTION_LINE_RE = re.compile(r"^(\s*ACTION\s+)(\S+)(.*)", re.MULTILINE)
_IF_LINE_RE = re.compile(
    r"^(\s*(?:ELSE\s+)?IF\s+\S+\s+)(\S+)(.*)", re.MULTILINE
)

# Handle/var property access: @prev.Name, $var.contents, @input.property
# Only match known DSL handle references (not URLs or other @-prefixed patterns)
_HANDLE_PROP_RE = re.compile(r"(@(?:prev|item|index|input|date|key|value))\.(\w+)")
_VAR_PROP_RE = re.compile(r"(\$\w+)\.(\w+)")

# Lazy singleton
_resolver: Optional[ActionResolver] = None


def _get_resolver() -> ActionResolver:
    global _resolver
    if _resolver is None:
        _resolver = ActionResolver()
    return _resolver


def _similarity(a: str, b: str) -> float:
    """Compute similarity score between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# ── Text-level repairs ──────────────────────────────────────────────

def _fix_multiline_interpolation(text: str, changes: list[LintChange]) -> str:
    """Collapse literal newlines inside backtick-delimited interpolated strings.

    The grammar's INTERP_TEXT terminal matches /[^`{}]+/ which does include
    newlines in Lark's default mode — but the model sometimes generates
    interpolated strings that break across DSL statement boundaries, causing
    the parser to misinterpret structural keywords as interpolation text.

    We detect this by finding backtick strings whose content contains newlines
    that look like they start new DSL statements (ACTION, SET, IF, etc.).
    Those newlines are replaced with \\n (escaped newline literal).
    """
    result = []
    i = 0
    in_backtick = False

    while i < len(text):
        if text[i] == '`' and (i == 0 or text[i-1] != '\\'):
            if not in_backtick:
                in_backtick = True
                result.append('`')
            else:
                in_backtick = False
                result.append('`')
            i += 1
        elif in_backtick and text[i] == '\n':
            # Check if the next non-whitespace after this newline looks like
            # a structural keyword — if so, the interpolation is broken
            rest = text[i+1:].lstrip()
            looks_structural = any(
                rest.startswith(kw) and (len(rest) == len(kw) or not rest[len(kw)].isalpha())
                for kw in _STRUCTURAL_KEYWORDS
                if len(rest) >= len(kw)
            )

            if looks_structural:
                # This newline breaks out of interpolation into structure.
                # Close the backtick here, let the structural content parse normally.
                result.append('`')
                result.append('\n')
                in_backtick = False
                line_num = text[:i].count('\n') + 1
                changes.append(LintChange(
                    line=line_num,
                    kind="interpolation",
                    original="<newline breaking interpolation>",
                    replacement="<closed backtick>",
                    confidence=0.85,
                ))
            else:
                # Newline inside interpolation that doesn't break structure —
                # replace with escaped newline to keep it in one logical line
                result.append('\\n')
                line_num = text[:i].count('\n') + 1
                changes.append(LintChange(
                    line=line_num,
                    kind="interpolation",
                    original="<literal newline in interpolation>",
                    replacement="\\\\n",
                    confidence=0.9,
                ))
            i += 1
        else:
            result.append(text[i])
            i += 1

    # If we ended inside a backtick, close it
    if in_backtick:
        result.append('`')
        line_num = text.count('\n') + 1
        changes.append(LintChange(
            line=line_num,
            kind="interpolation",
            original="<unclosed backtick>",
            replacement="<auto-closed>",
            confidence=0.8,
        ))

    return ''.join(result)


def _fix_handle_property_access(text: str, changes: list[LintChange]) -> str:
    """Strip property access from handles and variables.

    The model sometimes generates @prev.Name or $var.contents — the grammar
    only allows @prev and $var. We strip the .property suffix.

    We must be careful not to touch dotted identifiers in ACTION names or
    param values in quoted strings/JSON.
    """
    def _strip_handle_prop(match: re.Match) -> str:
        handle = match.group(1)
        prop = match.group(2)

        # Don't strip inside quoted strings — we operate on the whole text
        # but the regex only matches outside quotes (approximately).
        # The handle_ref grammar allows: @prev, @item, @index, @input, @date, @IDENT
        # If handle is one of these and prop follows, strip it.
        line_num = text[:match.start()].count("\n") + 1
        changes.append(LintChange(
            line=line_num,
            kind="handle",
            original=f"{handle}.{prop}",
            replacement=handle,
            confidence=0.85,
        ))
        return handle

    # Only strip property access when the handle/var appears outside quotes
    # We do a simplified approach: process each line, skip content in quotes
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        # Don't modify lines that are pure quoted strings or inside JSON
        # Simple heuristic: only fix @handle.prop and $var.prop patterns
        # that appear in parameter value positions
        fixed_line = _HANDLE_PROP_RE.sub(_strip_handle_prop, line)
        fixed_line = _VAR_PROP_RE.sub(
            lambda m: _strip_var_prop(m, text, changes),
            fixed_line,
        )
        fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines)


def _strip_var_prop(match: re.Match, text: str, changes: list[LintChange]) -> str:
    """Strip property suffix from $var.property."""
    var = match.group(1)
    prop = match.group(2)
    line_num = text[:match.start()].count("\n") + 1
    changes.append(LintChange(
        line=line_num,
        kind="handle",
        original=f"{var}.{prop}",
        replacement=var,
        confidence=0.85,
    ))
    return var


# ── Line-level repairs ──────────────────────────────────────────────

def _suffix_similarity(a: str, b: str) -> float:
    """Compute similarity between the differing suffixes of two dotted names.

    This prevents false matches where a long shared prefix (e.g.,
    'is.workflow.actions.') inflates the overall similarity score.
    """
    a_parts = a.split(".")
    b_parts = b.split(".")

    # Find the common prefix length
    common = 0
    for ap, bp in zip(a_parts, b_parts):
        if ap == bp:
            common += 1
        else:
            break

    # Compare only the differing suffixes
    a_suffix = ".".join(a_parts[common:])
    b_suffix = ".".join(b_parts[common:])

    if not a_suffix or not b_suffix:
        return _similarity(a, b)

    return _similarity(a_suffix, b_suffix)


def _fix_action_names(text: str, changes: list[LintChange]) -> str:
    """Fix hallucinated action names via multi-strategy fuzzy matching.

    Uses tiered confidence:
      - High-confidence aliases (HALLUCINATION_ALIASES): kind="action", confidence=0.95
      - Semantic-risky aliases (SEMANTIC_RISKY_ALIASES): kind="alias_warning", confidence=0.7
      - Fuzzy matches: kind="action", confidence=similarity score
    """
    resolver = _get_resolver()

    def _replace_action(match: re.Match) -> str:
        prefix = match.group(1)
        action_name = match.group(2)
        rest = match.group(3)

        # Skip 'comment' — always valid and very common
        if action_name == "comment":
            return match.group(0)

        if resolver.is_valid(action_name):
            return match.group(0)

        closest, is_alias, reason = resolver.find_closest(action_name, cutoff=0.65)
        if closest and closest != action_name:
            # Determine if this is a semantic-risky alias
            is_risky, risky_reason = resolver.is_semantic_risky(action_name)

            if not is_alias:
                # For fuzzy matches (not aliases), validate suffix similarity
                # to prevent "is.workflow.actions.X" matching "is.workflow.actions.Y"
                # when X and Y are completely different words
                suffix_sim = _suffix_similarity(action_name, closest)
                if suffix_sim < 0.55:
                    # Suffix too different — this is likely a wrong match
                    return match.group(0)

            # Tiered confidence and kind
            if is_risky:
                kind = "alias_warning"
                confidence = 0.7
                change_reason = reason or risky_reason
            elif is_alias:
                kind = "action"
                confidence = 0.95
                change_reason = reason
            else:
                kind = "action"
                confidence = _similarity(action_name, closest)
                change_reason = reason

            line_num = text[:match.start()].count("\n") + 1
            changes.append(LintChange(
                line=line_num,
                kind=kind,
                original=action_name,
                replacement=closest,
                confidence=confidence,
                reason=change_reason,
            ))
            return f"{prefix}{closest}{rest}"

        return match.group(0)

    return _ACTION_LINE_RE.sub(_replace_action, text)


def _fix_conditions(text: str, changes: list[LintChange]) -> str:
    """Fix invalid condition keywords in IF statements."""

    def _replace_condition(match: re.Match) -> str:
        prefix = match.group(1)
        condition = match.group(2)
        rest = match.group(3)

        if condition in VALID_CONDITIONS:
            return match.group(0)

        # Check known aliases first
        if condition in CONDITION_ALIASES:
            replacement = CONDITION_ALIASES[condition]
            line_num = text[:match.start()].count("\n") + 1
            changes.append(LintChange(
                line=line_num,
                kind="condition",
                original=condition,
                replacement=replacement,
                confidence=0.95,
            ))
            return f"{prefix}{replacement}{rest}"

        # Fuzzy match against valid conditions
        matches = get_close_matches(condition, VALID_CONDITIONS, n=1, cutoff=0.6)
        if matches:
            replacement = matches[0]
            line_num = text[:match.start()].count("\n") + 1
            changes.append(LintChange(
                line=line_num,
                kind="condition",
                original=condition,
                replacement=replacement,
                confidence=_similarity(condition, replacement),
            ))
            return f"{prefix}{replacement}{rest}"

        return match.group(0)

    return _IF_LINE_RE.sub(_replace_condition, text)


# ── ACTION-as-keyword rewriting ──────────────────────────────────────
# When the model writes ACTION menu / ACTION repeat / ACTION choosefrommenu,
# it's confusing action names with DSL structural keywords.
# These need to be rewritten to actual DSL keywords (MENU, REPEAT, FOREACH).

_ACTION_KEYWORD_MAP: dict[str, tuple[str, str | None]] = {
    # action_name_lower → (DSL_KEYWORD, param_key_to_extract_value_from)
    "menu": ("MENU", "WFMenuPrompt"),
    "choosefrommenu": ("MENU", "WFMenuPrompt"),
    "repeat": ("REPEAT", "WFRepeatCount"),
    "repeat_with_each": ("FOREACH", None),
    "foreach": ("FOREACH", None),
    "repeat.count": ("REPEAT", "WFRepeatCount"),
    "repeat.each": ("FOREACH", None),
}

# Pattern to extract a param value like Key="Value" or Key=123
_PARAM_VALUE_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*(\S+)')


def _extract_param_value(line: str, param_key: str) -> str | None:
    """Extract the value of a named parameter from an ACTION line.

    Handles both quoted and unquoted values:
      WFMenuPrompt="Choose option"  → "Choose option"
      WFRepeatCount=5               → "5"
    """
    for m in _PARAM_VALUE_RE.finditer(line):
        key = m.group(1) or m.group(3)
        value = m.group(2) if m.group(2) is not None else m.group(4)
        if key and key.lower() == param_key.lower():
            return value
    return None


def _fix_action_as_keyword(text: str, changes: list[LintChange]) -> str:
    """Rewrite ACTION lines that use DSL structural keywords as action names.

    Detects patterns like:
      ACTION menu WFMenuPrompt="Choose"     → MENU "Choose"
      ACTION repeat WFRepeatCount=5         → REPEAT 5
      ACTION repeat_with_each               → FOREACH @input
      ACTION choosefrommenu WFMenuPrompt=X  → MENU X
    """
    lines = text.split("\n")
    fixed = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith("ACTION "):
            fixed.append(line)
            continue

        # Extract the action name (second token)
        parts = stripped.split(None, 2)
        if len(parts) < 2:
            fixed.append(line)
            continue

        action_name = parts[1].lower()
        if action_name not in _ACTION_KEYWORD_MAP:
            fixed.append(line)
            continue

        keyword, param_key = _ACTION_KEYWORD_MAP[action_name]

        # Extract value from param if present
        value = None
        if param_key and len(parts) > 2:
            value = _extract_param_value(stripped, param_key)

        if value is not None:
            if keyword == "REPEAT":
                new_line = f"{keyword} {value}"
            else:
                new_line = f'{keyword} "{value}"'
        elif keyword == "FOREACH":
            new_line = f"{keyword} @input"
        elif keyword == "REPEAT":
            new_line = f"{keyword} 1"  # default repeat count
        else:
            new_line = keyword

        changes.append(LintChange(
            line=i + 1,
            kind="structure",
            original=stripped[:60],
            replacement=new_line,
            confidence=0.9,
            reason=f"ACTION used as DSL keyword: {action_name} → {keyword}",
        ))
        fixed.append(new_line)

    return "\n".join(fixed)


def _fix_incomplete_actions(text: str, changes: list[LintChange]) -> str:
    """Remove incomplete ACTION lines (ACTION without an action name)."""
    lines = text.split("\n")
    fixed_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # ACTION followed by nothing or just whitespace
        if stripped == "ACTION" or stripped == "ACTION ":
            changes.append(LintChange(
                line=i + 1,
                kind="structure",
                original=stripped,
                replacement="<removed>",
                confidence=0.9,
            ))
            continue
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _fix_truncated_lines(text: str, changes: list[LintChange]) -> str:
    """Fix lines that appear to be truncated mid-token.

    Common pattern: model hits max_tokens mid-line, producing something like:
      ACTION com.apple.MobileSMS.OpenConver
    (truncated identifier). If the last line doesn't end with a recognized
    terminal pattern, remove it.
    """
    lines = text.rstrip('\n').split('\n')
    if not lines:
        return text

    last_line = lines[-1].strip()

    # If the last line is empty or a valid closer/statement, it's fine
    if not last_line:
        return text

    # Check for obviously truncated lines: ends mid-identifier, mid-string, etc.
    # A valid DSL line ends with: quoted string, number, bool, `, ), ], }, IDENT, @IDENT, $IDENT,
    # or is a structural keyword (ENDIF, ENDMENU, etc.)
    structural_keywords = {'ENDIF', 'ENDMENU', 'ENDFOREACH', 'ENDREPEAT', 'ELSE'}

    if last_line in structural_keywords:
        return text

    # Check: does the last line look like it was cut off mid-way?
    # Heuristic: if it ends with an incomplete dotted identifier (no = or value after)
    # and has no parameter, it might be truncated
    if (last_line.startswith("ACTION ") and
            "=" not in last_line and
            last_line.count(" ") == 1):
        # ACTION with just a partial name and no params — possibly truncated
        # Check if the action name looks incomplete (doesn't resolve)
        action_candidate = last_line.split()[-1]
        resolver = _get_resolver()
        closest_trunc, _, _ = resolver.find_closest(action_candidate, cutoff=0.7)
        if not resolver.is_valid(action_candidate) and not closest_trunc:
            changes.append(LintChange(
                line=len(lines),
                kind="structure",
                original=last_line[:60],
                replacement="<removed truncated line>",
                confidence=0.7,
            ))
            lines = lines[:-1]

    return '\n'.join(lines) + ('\n' if text.endswith('\n') else '')


# ── Structural repairs ──────────────────────────────────────────────

def _fix_structure(text: str, changes: list[LintChange]) -> str:
    """Fix structural issues: unclosed blocks, truncated output.

    Tracks block nesting and auto-closes any unclosed blocks
    (from innermost to outermost) at the end of the DSL.
    """
    lines = text.split("\n")

    # Track open blocks with a stack for proper nesting order
    block_stack: list[str] = []  # stack of block types: "IF", "MENU", "FOREACH", "REPEAT"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("IF "):
            block_stack.append("IF")
        elif stripped == "ELSE" or stripped.startswith("ELSE IF "):
            pass  # ELSE doesn't open a new block
        elif stripped == "ENDIF":
            # Pop the most recent IF
            for j in range(len(block_stack) - 1, -1, -1):
                if block_stack[j] == "IF":
                    block_stack.pop(j)
                    break
        elif stripped.startswith("MENU ") or stripped == "MENU":
            block_stack.append("MENU")
        elif stripped == "ENDMENU":
            for j in range(len(block_stack) - 1, -1, -1):
                if block_stack[j] == "MENU":
                    block_stack.pop(j)
                    break
        elif stripped.startswith("FOREACH"):
            block_stack.append("FOREACH")
        elif stripped == "ENDFOREACH":
            for j in range(len(block_stack) - 1, -1, -1):
                if block_stack[j] == "FOREACH":
                    block_stack.pop(j)
                    break
        elif stripped.startswith("REPEAT"):
            block_stack.append("REPEAT")
        elif stripped == "ENDREPEAT":
            for j in range(len(block_stack) - 1, -1, -1):
                if block_stack[j] == "REPEAT":
                    block_stack.pop(j)
                    break

    # Close any unclosed blocks (pop from stack = innermost first)
    if block_stack:
        closers = []
        for block_type in reversed(block_stack):
            closer = f"END{block_type}"
            closers.append(closer)
            changes.append(LintChange(
                line=len(lines),
                kind="structure",
                original="<truncated>",
                replacement=closer,
                confidence=0.8,
            ))
        text = text.rstrip("\n") + "\n" + "\n".join(closers) + "\n"

    return text


def _fix_orphan_else(text: str, changes: list[LintChange]) -> str:
    """Remove ELSE clauses that appear outside any IF block.

    The model sometimes generates ELSE without a matching IF, or generates
    ELSE after ENDIF. These orphan ELSE clauses cause parse errors.
    """
    lines = text.split("\n")
    fixed_lines = []
    if_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("IF "):
            if_depth += 1
            fixed_lines.append(line)
        elif stripped == "ENDIF":
            if if_depth > 0:
                if_depth -= 1
            fixed_lines.append(line)
        elif stripped == "ELSE" or stripped.startswith("ELSE IF "):
            if if_depth > 0:
                # Valid ELSE inside an IF block
                fixed_lines.append(line)
            else:
                # Orphan ELSE — remove it
                changes.append(LintChange(
                    line=i + 1,
                    kind="structure",
                    original=stripped[:40],
                    replacement="<removed orphan ELSE>",
                    confidence=0.8,
                ))
                # Also remove any content until the next structural keyword or ENDIF
                continue
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


# ── Final repairs ───────────────────────────────────────────────────

def _fix_endshortcut(text: str, changes: list[LintChange]) -> str:
    """Handle ENDSHORTCUT: truncate ramble after it, or append if missing.

    Truncation guards:
      - Only treat ENDSHORTCUT at the START of a line as structural.
      - Track quote/backtick state to avoid truncating if ENDSHORTCUT
        appears inside a quoted string or backtick interpolation.
    """
    lines = text.split("\n")
    in_quoted = False
    in_backtick = False
    endshortcut_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Update quote/backtick state (simple heuristic: count unescaped delimiters)
        for ch_idx, ch in enumerate(stripped):
            if ch == '"' and (ch_idx == 0 or stripped[ch_idx - 1] != '\\'):
                in_quoted = not in_quoted
            elif ch == '`' and (ch_idx == 0 or stripped[ch_idx - 1] != '\\'):
                in_backtick = not in_backtick

        # Only line-start ENDSHORTCUT outside strings is structural
        if stripped == "ENDSHORTCUT" and not in_quoted and not in_backtick:
            endshortcut_idx = i
            break

    if endshortcut_idx is not None:
        # Truncate everything after ENDSHORTCUT
        remaining = [l for l in lines[endshortcut_idx + 1:] if l.strip()]
        if remaining:
            changes.append(LintChange(
                line=endshortcut_idx + 2,
                kind="structure",
                original=f"<{len(remaining)} lines after ENDSHORTCUT>",
                replacement="<truncated>",
                confidence=0.95,
            ))
        text = "\n".join(lines[:endshortcut_idx + 1]) + "\n"
    else:
        # ENDSHORTCUT absent — auto-append
        text = text.rstrip("\n") + "\nENDSHORTCUT\n"
        changes.append(LintChange(
            line=text.count("\n"),
            kind="structure",
            original="<missing>",
            replacement="ENDSHORTCUT",
            confidence=1.0,
        ))

    return text


def _fix_trailing_newline(text: str, changes: list[LintChange]) -> str:
    """Ensure text ends with a newline (grammar requires it)."""
    if not text.endswith("\n"):
        changes.append(LintChange(
            line=text.count("\n") + 1,
            kind="trailing_newline",
            original="<no newline>",
            replacement="\\n",
            confidence=1.0,
        ))
        text += "\n"
    return text


def _strip_markdown_fences(text: str, changes: list[LintChange]) -> str:
    """Strip markdown code fences if present.

    LLMs sometimes wrap DSL output in ```dsl ... ``` or similar.
    """
    stripped = text.strip()
    fence_re = re.compile(r"^```\w*\s*\n(.*?)```\s*$", re.DOTALL)
    m = fence_re.match(stripped)
    if m:
        inner = m.group(1).strip()
        if inner.startswith("SHORTCUT"):
            changes.append(LintChange(
                line=1,
                kind="structure",
                original="```...```",
                replacement="<unwrapped>",
                confidence=1.0,
            ))
            return inner
    return text


def _strip_preamble(text: str, changes: list[LintChange]) -> str:
    """Strip any preamble text before SHORTCUT declaration.

    LLMs sometimes emit explanation text before the actual DSL.
    """
    text = text.strip()
    if not text.startswith("SHORTCUT"):
        m = re.search(r'(SHORTCUT\s+".*)', text, re.DOTALL)
        if m:
            preamble = text[:m.start()].strip()
            if preamble:
                changes.append(LintChange(
                    line=1,
                    kind="structure",
                    original=preamble[:40] + "..." if len(preamble) > 40 else preamble,
                    replacement="<stripped preamble>",
                    confidence=0.9,
                ))
            text = m.group(1)
    return text


# ── Phase 0: Macro Expansion ─────────────────────────────────────────

_macro_expander: MacroExpander | None = None


def _expand_macros(text: str, changes: list[LintChange]) -> str:
    """Phase 0: Expand MACRO directives into multi-action DSL sequences.

    Runs BEFORE text extraction and all other lint phases. Expanded text
    goes through the full pipeline: lint → parse → validate → compile.

    MACRO directives in the text are replaced with their expansion templates
    from macro_patterns.json. Change records are logged with kind="macro_expansion".
    """
    global _macro_expander
    if _macro_expander is None:
        _macro_expander = MacroExpander()

    expander: MacroExpander = _macro_expander  # type: ignore[assignment]
    expanded, expansions = expander.expand(text)

    for exp in expansions:
        changes.append(LintChange(
            line=exp.line,
            kind="macro_expansion",
            original=exp.original[:60] + "..." if len(exp.original) > 60 else exp.original,
            replacement=exp.expanded[:60] + "..." if len(exp.expanded) > 60 else exp.expanded,
            confidence=1.0,
            reason=f"Expanded macro: {exp.macro_name} ({exp.param_values})",
        ))

    return expanded


# ── Main Entry Point ─────────────────────────────────────────────────

def lint_dsl(text: str) -> LintResult:
    """Lint and repair raw DSL text from LLM output.

    Applies repairs in a carefully ordered pipeline:

      Phase 0 — Macro expansion:
        0. Expand MACRO directives into multi-action DSL sequences

      Phase 1 — Text extraction:
        1. Strip markdown fences
        2. Strip preamble text

      Phase 2 — Text-level repairs (before line parsing):
        3. Fix multi-line interpolation (collapse newlines in backtick strings)

      Phase 3 — Line-level repairs:
        4. Fix incomplete/empty ACTION lines
        5. Fix hallucinated action names (fuzzy match to catalog)
        6. Fix invalid condition keywords (alias table + fuzzy)
        7. Fix handle/variable property access (@prev.Name → @prev)

      Phase 4 — Structural repairs:
        8. Fix orphan ELSE clauses
        9. Fix truncated last lines
        10. Fix unclosed blocks (auto-close from truncation)

      Phase 5 — Final:
        11. Ensure trailing newline

    Returns LintResult with fixed text and list of changes made.
    """
    changes: list[LintChange] = []

    # Phase 0: Macro expansion
    text = _expand_macros(text, changes)

    # Phase 1: Text extraction
    text = _strip_markdown_fences(text, changes)
    text = _strip_preamble(text, changes)

    # Phase 2: Text-level repairs
    text = _fix_multiline_interpolation(text, changes)

    # Phase 3: Line-level repairs
    text = _fix_incomplete_actions(text, changes)
    text = _fix_action_as_keyword(text, changes)
    text = _fix_action_names(text, changes)
    text = _fix_conditions(text, changes)
    text = _fix_handle_property_access(text, changes)

    # Phase 4: Structural repairs
    text = _fix_orphan_else(text, changes)
    text = _fix_truncated_lines(text, changes)
    text = _fix_structure(text, changes)
    text = _fix_endshortcut(text, changes)

    # Phase 5: Final
    text = _fix_trailing_newline(text, changes)

    return LintResult(text=text, changes=changes)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result = lint_dsl(text)
    print(result.text)

    if result.changes:
        print(f"\n--- {len(result.changes)} repairs made ---", file=sys.stderr)
        for c in result.changes:
            reason_str = f" — {c.reason}" if c.reason else ""
            print(
                f"  L{c.line} [{c.kind}] {c.original!r} → {c.replacement!r} "
                f"(confidence={c.confidence:.2f}){reason_str}",
                file=sys.stderr,
            )
