# ShortcutForge

**Describe what you want your phone to do. Get a signed, installable Apple Shortcut.**

---

ShortcutForge is a natural language compiler targeting the Apple Shortcuts runtime — the automation layer that ships on every iPhone, iPad, and Mac. You describe what you want; it builds you a real program. Not a prototype, not a suggestion — a signed `.shortcut` binary that runs natively with full system permissions.

HealthKit logging. REST API integrations. HomeKit automations. File management workflows. A 50-action, 6-level-deep monstrosity that blasts a specific frequency through your speaker to eject water from the grille. If Shortcuts can express it, ShortcutForge can compile it from a sentence.

```
"Log my caffeine intake to Apple Health"
  → ShortcutDSL (human-readable IR)
    → 6-phase deterministic linter
      → LALR(1) parser → typed IR
        → semantic validation (615-action catalog)
          → static analysis (7 passes)
            → plist compiler + signing
              → installable .shortcut file
```

## Why This Exists

Someone kept forgetting to log their meds, wished they could describe what they wanted and have it parsed into Apple Health, and then — in the space of about 24 hours — accidentally built a compiler for the entire Apple platform.

The original problem was 12 lines of DSL. The solution became 21 compiler modules, a 615-action catalog, a 6-phase deterministic linter, 7 static analyses, a reverse compiler, a distillation pipeline, and a model training loop. These things happen.

## The Big Idea

Apple Shortcuts has hooks into *everything*: HealthKit, HomeKit, Calendar, Contacts, Files, Photos, Maps, Messages, Mail, Safari, Siri, accessibility APIs, hardware controls, and every app that exposes Shortcut actions. Combined with an unrestricted HTTP client (`downloadurl`), that's any REST API on earth — including serverless backends you can stand up on the fly.

ShortcutForge turns all of that into a natural language compilation target:

```
"arbitrary human intent"
  → compiler pipeline (on-device or cloud)
    → verified, signed, installable program
      → the entire Apple system surface
        → plus any API endpoint on the internet
```

This isn't an app. It's a programmable bridge between natural language and full device-level access, running on a $400 phone with no developer account, no jailbreak, no sideloading. The output is the exact same artifact a human would produce in the Shortcuts editor — signed through Apple's own infrastructure.

And the attack surface (in the good sense) grows every WWDC. Every action Apple adds to Shortcuts automatically expands the expressible program space without any changes to ShortcutForge.

## The Secret: LLMs Are Unreliable, Compilers Aren't

LLM output is not trusted. It's treated as *untrusted intermediate code* and fed through a real compiler pipeline. The model doesn't need to be perfect. It just needs to land close enough for the compiler infrastructure to take over.

The 6-phase deterministic linter is the unsung hero — it turns "almost right" into "actually right" before the parser ever sees it. Hallucinated action names get fuzzy-matched against the 615-action catalog. Unclosed blocks get repaired. Invalid conditions get normalized. The linter isn't cleaning up after the model; it's the *primary reliability mechanism*. The model is just the front-end.

This means a **tiny model works**. The structural work — action selection, parameter wiring, control flow — is a constrained vocabulary problem: 615 actions, a finite grammar, deterministic compilation. A 0.5B–1B parameter model can handle the common case. Creative decomposition ("how do I eject water with no water action?") is the rare path where a frontier model earns its keep.

## Dual Intelligence

| Complexity | Handler | Latency | Cost |
|---|---|---|---|
| Simple toggle | 0.5B on-device | ~2s | Free |
| Medium workflow | 1B on-device | ~5s | Free |
| Complex multi-domain | 8B local | ~15s | Free |
| Truly creative | Claude via MCP | ~30s | Fractions of a cent |

For the ambitious case, ShortcutForge exposes its full pipeline as an [MCP server](cli/mcp_server.py) with 27 tool endpoints. A frontier model doesn't generate and pray — it *uses the compiler as a development environment*, iterating through lint → validate → analyze → compile the same way a human developer would, but in 30 seconds across 6 tool calls.

## Quick Start

```bash
# Install
git clone https://github.com/rohanvinaik/ShortcutForge.git
cd ShortcutForge
pip install -e .

# Generate with Claude API
export ANTHROPIC_API_KEY=sk-...
python cli/shortcutforge.py "Set a 5-minute timer for tea"

# Generate with local model (Apple Silicon)
pip install -e ".[local]"
python cli/shortcutforge.py "Toggle Do Not Disturb" \
  --engine local \
  --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/baseline-v1-mlx

# Compile DSL directly (no LLM needed)
python cli/shortcutforge.py --dsl-file my_shortcut.dsl

# Web UI
python cli/server.py
# → http://localhost:8000

# MCP server (for Claude Code / agentic workflows)
python cli/mcp_server.py
```

> **Prerequisite (one-time):** On your Mac/iPhone, enable Settings → Shortcuts → Private Sharing. This allows importing signed shortcuts.

## What People Build With This

These aren't hypothetical. They're real shortcuts from a corpus of 1,772 production Apple Shortcuts that informed ShortcutForge's design:

- **Water ejection** — plays a specific-frequency audio tone at max volume to vibrate water from speaker grilles. No "eject water" action exists. The insight: timed hardware state changes create physical-world effects.
- **Morse code flashlight** — converts text to dots/dashes, pulses the flashlight with computed delays.
- **AI facial recognition** (34 actions) — sends photos to DeepAI, parses bounding box coordinates, overlays visual markers.
- **Medical logging system** (795 actions) — multi-category health tracking with file-persisted JSON database, 11 loops, menu-driven CRUD. Depth 8 nesting.
- **Sleep analytics with Charty charts** (1,253 actions) — queries HealthKit for sleep data, computes statistics, renders visual charts via third-party intents. The largest shortcut in the corpus.
- **AI personal assistant with tools** (712 actions) — natural-language-driven, calls 9 sub-shortcuts as "tools," self-updates, logs to Console.
- **Shortcut merger/editor** (850 actions) — treats shortcuts as data, parsing plist XML with text processing chains, modifying action lists, reassembling. Meta-programming.
- **Full OAuth2 flow** (176 actions) — generates auth URL, handles callback, exchanges code for token, refreshes, persists credentials. 9 levels of nesting.
- **Batch birthday messages with AI** (52 actions) — selects contacts, feeds each name to Apple Intelligence, auto-sends personalized iMessages.

The creative ceiling isn't defined by what actions exist. It's defined by how you combine them.

## Architecture

### The Pipeline

```
Prompt (natural language)
  │
  ├─ Execution Planner ────── archetype classification + structured plan
  ├─ Architecture Reasoner ── shortcut-only vs. hybrid (needs server?)
  ├─ Scenario Profile ─────── domain + creative mode + token budget
  ├─ Domain Profile ────────── action context + validation rules
  ├─ Snippet Retrieval ─────── top-3 micro-patterns from registry
  │
  ▼
LLM Generation (Claude API or local MLX model)
  │
  ▼
DSL Linter / Canonicalizer (6 phases)
  ├─ Phase 0: Macro expansion (31 templates → multi-action DSL)
  ├─ Phase 1: Text extraction (markdown fences, prose stripping)
  ├─ Phase 2: Text-level repairs (interpolation, aliases)
  ├─ Phase 3: Line-level repairs (action name fuzzy resolution)
  ├─ Phase 4: Structural repairs (block closure, ENDSHORTCUT)
  └─ Phase 5: Final normalization
  │
  ▼
Lark LALR(1) Parser → ShortcutIR (typed dataclasses)
  │
  ▼
Semantic Validator
  ├─ 4-tier action resolution (known → alias → vendor-prefix → unknown)
  ├─ Strict / permissive dual modes
  └─ Domain-aware rules (HealthKit units, API error handling, etc.)
  │
  ▼
Simulation Harness (7 static analyses)
  ├─ Variable flow (set-before-use, branch coverage)
  ├─ Loop bound checking
  ├─ Menu case completeness
  ├─ Dead code detection
  ├─ API endpoint validation
  ├─ Type flow tracking
  └─ Contract validation (13 rules)
  │
  ▼
Compiler Bridge → Plist Compiler → Apple Signing → .shortcut
```

### Failure-Type-Routed Retry

No shotgun retries. Each failure type has a specific remedy:

| Failure | Retry Strategy |
|---|---|
| Token overflow | Escalate budget (simple → medium → complex → very_complex) |
| Parse syntax | Engage grammar constraints via Outlines |
| Unknown action | Feed validation errors back to LLM |
| Compile error | No retry (deterministic — if compilation fails, the IR is wrong) |

### The Data Flywheel

The system generates its own training data as a side effect of normal operation:

1. **Reverse compiler** decompiles 1,772 real Apple Shortcuts into validated DSL → training pairs
2. **Distillation logging** captures `(raw_output → canonicalized_output)` pairs from every generation
3. **Hard-negative mining** extracts examples where the linter saved broken output — the most informative training signal
4. **Quality curation** filters by parse+validate+compile, deduplicates, balances across domains

The linter repairs aren't wasted work. They're *free labeled data* for the next training run.

## ShortcutDSL

A line-oriented DSL designed for both humans and LLMs. Maps 1:1 to Apple Shortcuts actions while being readable, lintable, and grammar-constrainable.

```text
SHORTCUT "NIH Food Lookup"

ACTION ask WFAskActionPrompt="What did you eat?" WFInputType="Text"
SET $food = @prev
ACTION url WFURLActionURL=`https://api.nal.usda.gov/fdc/v1/foods/search?query={$food}&api_key=DEMO_KEY`
ACTION downloadurl WFHTTPMethod="GET"
ACTION detect.dictionary
ACTION getvalueforkey WFDictionaryKey="foods"
ACTION getitemfromlist WFItemSpecifier="First Item"
ACTION getvalueforkey WFDictionaryKey="description"
ACTION showresult Text=`Found: {@prev}`

ENDSHORTCUT
```

The same Lark LALR(1) grammar that *parses* the output can also *constrain* the output via [Outlines](https://github.com/dottxt-ai/outlines). The grammar that checks correctness *is* the grammar that enforces correctness — a closed loop where the DSL is its own specification and its own guardrail.

## Current Metrics

Frozen eval set: 100 examples. Baseline: Llama 3.1 8B (4-bit) + LoRA. Linter v2.4.

| Metric | Strict | Permissive |
|---|---|---|
| Parse | 93% | — |
| Validate | 85% | 89% |
| Compile | 85% | 89% |

22 standalone test suites. Zero regressions against frozen baseline.

## Project Structure

```
ShortcutForge/
├── src/                    # Core compiler pipeline (21 modules)
│   ├── orchestrator.py     #   Central pipeline: generate → lint → parse → validate → compile
│   ├── dsl_linter.py       #   6-phase deterministic linter (v2.4)
│   ├── dsl_parser.py       #   Lark LALR(1) parser
│   ├── dsl_validator.py    #   Semantic validator (615-action catalog)
│   ├── simulation_harness.py   7 static analyses
│   ├── shortcuts_compiler.py   Plist compiler + signing
│   └── ...                 #   + 15 more modules (planner, reasoner, profiles, etc.)
│
├── cli/                    # Entry points
│   ├── shortcutforge.py    #   CLI: "make me a shortcut"
│   ├── server.py           #   Web UI with SSE streaming
│   └── mcp_server.py       #   MCP server (27 tools)
│
├── tests/                  # 22 standalone test suites
├── training/               # ML training, eval, distillation (12 files)
├── research/               # Balanced Sashimi architecture research
├── tools/                  # One-off utilities and scrapers
│
├── references/
│   ├── shortcutdsl.lark    #   The grammar
│   ├── action_catalog.json #   615 actions + 659 aliases
│   ├── macro_patterns.json #   31 macro templates
│   ├── snippet_registry.json   ~200 micro-patterns
│   └── scenario_packs/     #   8 benchmark scenarios with rubrics
│
├── training_data/          # JSONL datasets + baselines
├── models/                 # Fine-tuned LoRA adapters (MLX)
├── downloaded/             # 1,772 real Apple Shortcuts corpus
├── output/                 # Generated .shortcut files
└── docs/                   # Documentation
```

## Research: Balanced Sashimi

The active research direction: a hybrid continuous-ternary architecture that decomposes generation into specialized modules instead of relying on a monolithic transformer.

The hypothesis is that the structural part of shortcut generation (picking the right actions and wiring them together) has low intrinsic dimensionality and can be handled by a ternary-weight decoder ({-1, 0, +1}), while only the free-text parts (user-facing strings, URLs, numeric values) need continuous precision.

Phase 0 (data preparation) is complete. Phase 1 (component validation) and Phase 2 (decoder training) are next. See [`research/docs/BALANCED_SASHIMI_RESEARCH.md`](research/docs/BALANCED_SASHIMI_RESEARCH.md) for the theory and [`research/README.md`](research/README.md) for current status.

## Data Snapshot

| Resource | Count |
|---|---|
| Actions in catalog | 615 |
| Canonical aliases | 659 |
| Macros | 31 (8 categories) |
| Real shortcuts (corpus) | 1,772 |
| Training examples | 6,679 |
| Frozen eval examples | 100 |
| Domain profiles | 8 |
| Scenario packs | 8 |
| Micro-pattern snippets | ~200 |
| Contract validation rules | 13 |

## License

Research project. Not affiliated with Apple.
