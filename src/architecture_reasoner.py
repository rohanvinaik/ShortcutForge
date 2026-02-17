"""
ShortcutForge Architecture Reasoner.

Analyzes prompts for hybrid signals that require external infrastructure
(server, database, webhook, etc.) beyond what a standalone Apple Shortcut
can provide.

Returns an ArchitectureDecision with:
  - strategy: "shortcut_only" (default) or "shortcut_plus_blueprint"
  - reason: human-readable explanation
  - blueprint_scope: what external components are needed (if hybrid)
  - hybrid_signals: detected hybrid indicators

For hybrid prompts, generates a blueprint document (not code) describing
the external component alongside the shortcut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

__version__ = "1.0"


# ── Hybrid Signal Keywords ─────────────────────────────────────────

# Keywords that suggest the shortcut needs an external component
_HYBRID_SIGNALS: dict[str, list[str]] = {
    "server": [
        "server", "backend", "deploy", "hosting", "cloud",
        "cloudflare worker", "aws lambda", "firebase", "vercel",
        "railway", "heroku",
    ],
    "database": [
        "database", "persistent storage", "store data permanently",
        "save to cloud", "sql", "mongodb", "redis", "supabase",
        "d1", "planetscale",
    ],
    "webhook": [
        "webhook", "push notification server", "callback url",
        "receive from", "listen for",
    ],
    "authentication": [
        "oauth flow", "login server", "auth server", "jwt",
        "session management", "user accounts",
    ],
    "realtime": [
        "websocket", "real-time sync", "live updates",
        "streaming", "server-sent events",
    ],
    "processing": [
        "image processing server", "ml model", "ai processing",
        "gpu processing", "heavy computation",
    ],
}

# Keywords that indicate shortcut-only is sufficient
_SHORTCUT_ONLY_SIGNALS = [
    "timer", "alarm", "reminder", "clipboard", "share", "menu",
    "brightness", "volume", "wallpaper", "bluetooth", "wifi",
    "airplane", "flashlight", "open app", "launch app",
    "photo", "camera", "text message", "email", "note",
    "calendar", "music", "speak", "vibrate", "notification",
]


# ── Data Classes ──────────────────────────────────────────────────

@dataclass
class ArchitectureDecision:
    """Result of architecture analysis for a prompt."""
    strategy: str  # "shortcut_only" or "shortcut_plus_blueprint"
    reason: str
    blueprint_scope: list[str] = field(default_factory=list)
    hybrid_signals: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def is_hybrid(self) -> bool:
        return self.strategy == "shortcut_plus_blueprint"


@dataclass
class BlueprintDoc:
    """A blueprint document for an external component."""
    title: str
    description: str
    components: list[str]
    integration_notes: str = ""


# ── Architecture Reasoner ────────────────────────────────────────

class ArchitectureReasoner:
    """Analyzes prompts to determine if external infrastructure is needed.

    Most prompts → shortcut_only (default, no changes to pipeline).
    Hybrid prompts → shortcut_plus_blueprint (generates blueprint doc).
    """

    def analyze(self, prompt: str) -> ArchitectureDecision:
        """Analyze a prompt for hybrid architecture signals.

        Returns an ArchitectureDecision with the recommended strategy.
        """
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\w+', prompt_lower))

        # Check for hybrid signals
        detected_signals: dict[str, list[str]] = {}
        for category, keywords in _HYBRID_SIGNALS.items():
            matches = []
            for kw in keywords:
                if " " in kw:
                    # Phrase match
                    if kw in prompt_lower:
                        matches.append(kw)
                else:
                    # Word match
                    if kw in prompt_words:
                        matches.append(kw)
            if matches:
                detected_signals[category] = matches

        # Check for shortcut-only signals
        shortcut_only_count = sum(
            1 for sig in _SHORTCUT_ONLY_SIGNALS
            if sig in prompt_lower
        )

        # Decision logic
        hybrid_score = sum(len(v) for v in detected_signals.values())

        if hybrid_score == 0:
            return ArchitectureDecision(
                strategy="shortcut_only",
                reason="No hybrid infrastructure signals detected",
                confidence=1.0,
            )

        if hybrid_score >= 3 or len(detected_signals) >= 2:
            # Strong hybrid signals
            blueprint_scope = list(detected_signals.keys())
            return ArchitectureDecision(
                strategy="shortcut_plus_blueprint",
                reason=f"Detected hybrid signals: {', '.join(f'{k}({len(v)})' for k, v in detected_signals.items())}",
                blueprint_scope=blueprint_scope,
                hybrid_signals=detected_signals,
                confidence=0.9,
            )

        if hybrid_score >= 1 and shortcut_only_count == 0:
            # Moderate hybrid signals, no shortcut-only counterweight
            blueprint_scope = list(detected_signals.keys())
            return ArchitectureDecision(
                strategy="shortcut_plus_blueprint",
                reason=f"Moderate hybrid signals: {', '.join(detected_signals.keys())}",
                blueprint_scope=blueprint_scope,
                hybrid_signals=detected_signals,
                confidence=0.6,
            )

        # Weak hybrid signals with shortcut-only counterweight
        return ArchitectureDecision(
            strategy="shortcut_only",
            reason=f"Weak hybrid signals ({hybrid_score}) overridden by shortcut-only signals ({shortcut_only_count})",
            hybrid_signals=detected_signals,
            confidence=0.7,
        )

    def generate_blueprint(self, decision: ArchitectureDecision, prompt: str) -> BlueprintDoc | None:
        """Generate a blueprint document for hybrid architectures.

        Returns None for shortcut_only decisions.
        """
        if not decision.is_hybrid:
            return None

        components = []
        integration_notes_parts = []

        for scope in decision.blueprint_scope:
            if scope == "server":
                components.append("Backend API server (e.g., Cloudflare Worker, AWS Lambda)")
                integration_notes_parts.append(
                    "The shortcut uses downloadurl to communicate with the server endpoint. "
                    "The server handles data processing and returns JSON responses."
                )
            elif scope == "database":
                components.append("Database or persistent storage (e.g., D1, KV, Supabase)")
                integration_notes_parts.append(
                    "Data persistence is handled by the server-side database. "
                    "The shortcut sends data via HTTP POST and retrieves via HTTP GET."
                )
            elif scope == "webhook":
                components.append("Webhook receiver endpoint")
                integration_notes_parts.append(
                    "The server exposes a webhook endpoint that receives push events. "
                    "The shortcut can register for these events or poll for updates."
                )
            elif scope == "authentication":
                components.append("Authentication service")
                integration_notes_parts.append(
                    "User authentication is handled server-side. "
                    "The shortcut stores auth tokens securely and includes them in API requests."
                )
            elif scope == "realtime":
                components.append("Real-time communication layer")
                integration_notes_parts.append(
                    "Real-time features require server-side support. "
                    "The shortcut polls for updates or uses a notification-based approach."
                )
            elif scope == "processing":
                components.append("Processing backend (ML, image, compute)")
                integration_notes_parts.append(
                    "Heavy processing is offloaded to a server-side backend. "
                    "The shortcut uploads data and retrieves processed results."
                )

        return BlueprintDoc(
            title=f"Blueprint: External Components for Shortcut",
            description=f"This shortcut requires external infrastructure to fully function. Prompt: {prompt[:100]}",
            components=components,
            integration_notes=" ".join(integration_notes_parts),
        )


# ── Convenience ──────────────────────────────────────────────────

def analyze_architecture(prompt: str) -> ArchitectureDecision:
    """Convenience function to analyze a prompt's architecture needs."""
    return ArchitectureReasoner().analyze(prompt)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Set a timer for 5 minutes"
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze(prompt)

    print(f"Prompt: {prompt!r}")
    print(f"Strategy: {decision.strategy}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence:.2f}")
    if decision.hybrid_signals:
        print(f"Hybrid signals: {decision.hybrid_signals}")
    if decision.blueprint_scope:
        print(f"Blueprint scope: {decision.blueprint_scope}")
        bp = reasoner.generate_blueprint(decision, prompt)
        if bp:
            print(f"\nBlueprint: {bp.title}")
            for comp in bp.components:
                print(f"  - {comp}")
            print(f"Integration: {bp.integration_notes[:100]}...")
