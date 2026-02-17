"""
ShortcutDSL Domain Profile Manager.

Manages domain profiles that inject specialized knowledge into the
generation pipeline. Profiles contain:
  - Keywords for automatic profile selection
  - Prompt context for domain-specific guidance
  - Relevant actions for the domain
  - Domain-specific data (e.g., HealthKit sample types, HTTP patterns)

Profiles are loaded from references/domain_profiles/*.json.
Profile selection is based on keyword matching against the user prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__version__ = "1.0"

# ── Paths ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_REFS_DIR = _SCRIPT_DIR.parent / "references"
_PROFILES_DIR = _REFS_DIR / "domain_profiles"

# ── Data Classes ───────────────────────────────────────────────────────

@dataclass
class DomainProfile:
    """A loaded domain profile."""
    profile_id: str
    keywords: list[str] = field(default_factory=list)
    prompt_context: str = ""
    relevant_actions: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        """Whether this profile has any prompt context to inject."""
        return bool(self.prompt_context.strip())

    def format_relevant_actions(self, catalog: dict | None = None) -> str:
        """Format relevant actions as a text block for prompt injection."""
        if not self.relevant_actions:
            return ""

        if catalog:
            # Format with descriptions from catalog
            actions_data = catalog.get("actions", {})
            canonical = catalog.get("_meta", {}).get("canonical_map", {})
            lines = []
            for short_name in self.relevant_actions:
                full_id = canonical.get(short_name, f"is.workflow.actions.{short_name}")
                entry = actions_data.get(full_id, {})
                desc = entry.get("description", "").strip()[:60]
                if desc:
                    lines.append(f"  {short_name}: {desc}")
                else:
                    lines.append(f"  {short_name}")
            return "\n".join(lines)
        else:
            # Simple list without descriptions
            return "\n".join(f"  {a}" for a in self.relevant_actions)


# ── Profile Manager ───────────────────────────────────────────────────

class DomainProfileManager:
    """Manages domain profiles for context-aware generation.

    Loads profiles from references/domain_profiles/*.json.
    Selects the best-matching profile for a given prompt based on keyword scoring.
    """

    def __init__(self, profiles_dir: Path | None = None):
        self._profiles_dir = profiles_dir or _PROFILES_DIR
        self._profiles: dict[str, DomainProfile] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load profiles on first access."""
        if self._loaded:
            return

        if not self._profiles_dir.is_dir():
            self._loaded = True
            return

        for path in self._profiles_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                meta = data.get("_meta", {})
                profile_id = meta.get("profile_id", path.stem)

                self._profiles[profile_id] = DomainProfile(
                    profile_id=profile_id,
                    keywords=data.get("keywords", []),
                    prompt_context=data.get("prompt_context", ""),
                    relevant_actions=data.get("relevant_actions", []),
                    data=data,
                )
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed profile files
                pass

        self._loaded = True

    def list_profiles(self) -> list[str]:
        """List all available profile IDs."""
        self._ensure_loaded()
        return list(self._profiles.keys())

    def get_profile(self, profile_id: str) -> Optional[DomainProfile]:
        """Get a specific profile by ID."""
        self._ensure_loaded()
        return self._profiles.get(profile_id)

    def select_profile(self, prompt: str) -> DomainProfile:
        """Select the best-matching domain profile for a prompt.

        Scores each profile by counting keyword matches in the prompt.
        Returns 'general' profile if no strong match found.

        Args:
            prompt: The user's natural language prompt

        Returns:
            The best-matching DomainProfile
        """
        self._ensure_loaded()

        if not self._profiles:
            return DomainProfile(profile_id="general")

        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\w+', prompt_lower))

        best_id = "general"
        best_score = 0

        for pid, profile in self._profiles.items():
            if pid == "general":
                continue  # General is the fallback, not scored

            score = 0
            for keyword in profile.keywords:
                kw_lower = keyword.lower()
                # Multi-word keyword: check if the phrase appears in the prompt
                if " " in kw_lower:
                    if kw_lower in prompt_lower:
                        score += 3  # Phrase matches are worth more
                else:
                    if kw_lower in prompt_words:
                        score += 1

            if score > best_score:
                best_score = score
                best_id = pid

        # Require at least 2 keyword matches to select a non-general profile
        if best_score < 2:
            best_id = "general"

        return self._profiles.get(best_id, DomainProfile(profile_id="general"))

    def get_prompt_context(self, profile_id: str) -> str:
        """Get the prompt context string for a profile."""
        self._ensure_loaded()
        profile = self._profiles.get(profile_id)
        if profile:
            return profile.prompt_context
        return ""

    def get_validation_data(self, profile_id: str) -> dict:
        """Get domain-specific validation data for a profile.

        Returns data relevant for domain-aware validation rules:
          - HealthKit: sample types, unit compatibility, value ranges
          - API: HTTP patterns, auth patterns
        """
        self._ensure_loaded()
        profile = self._profiles.get(profile_id)
        if not profile:
            return {}

        data = profile.data
        result: dict[str, Any] = {}

        # Health-specific validation data
        if "hk_sample_types" in data:
            result["hk_sample_types"] = data["hk_sample_types"]
        if "unit_compatibility" in data:
            result["unit_compatibility"] = data["unit_compatibility"]
        if "value_ranges" in data:
            result["value_ranges"] = data["value_ranges"]

        # API-specific validation data
        if "http_patterns" in data:
            result["http_patterns"] = data["http_patterns"]
        if "auth_patterns" in data:
            result["auth_patterns"] = data["auth_patterns"]

        return result


# ── Convenience Functions ─────────────────────────────────────────────

_manager: DomainProfileManager | None = None


def get_domain_profile_manager(profiles_dir: Path | None = None) -> DomainProfileManager:
    """Get the global domain profile manager (singleton)."""
    global _manager
    if _manager is None or profiles_dir is not None:
        _manager = DomainProfileManager(profiles_dir)
    return _manager


def select_domain_profile(prompt: str) -> DomainProfile:
    """Select the best domain profile for a prompt (convenience function)."""
    return get_domain_profile_manager().select_profile(prompt)


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mgr = DomainProfileManager()
    profiles = mgr.list_profiles()
    print(f"Available profiles ({len(profiles)}): {', '.join(profiles)}\n")

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        profile = mgr.select_profile(prompt)
        print(f"Prompt: {prompt!r}")
        print(f"Selected profile: {profile.profile_id}")
        if profile.has_context:
            print(f"Context: {profile.prompt_context[:100]}...")
        if profile.relevant_actions:
            print(f"Relevant actions: {', '.join(profile.relevant_actions[:10])}")
    else:
        for pid in profiles:
            profile = mgr.get_profile(pid)
            if profile:
                print(f"  {pid}:")
                print(f"    Keywords: {profile.keywords[:5]}...")
                print(f"    Context: {profile.prompt_context[:60]}..." if profile.prompt_context else "    Context: (none)")
                print(f"    Actions: {len(profile.relevant_actions)}")
