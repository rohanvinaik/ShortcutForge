# Documentation

## Guides

| Document | Audience | Description |
|----------|----------|-------------|
| [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md) | Developers, LLM agents | Quick reference for the lint → validate → compile pipeline. Includes MCP tool usage and the direct-Python workaround for large files. |
| [DSL_PATTERNS.md](DSL_PATTERNS.md) | Developers, LLM agents | Copy-paste ShortcutDSL patterns: API calls, JSON parsing, HealthKit logging, regex extraction, accumulators, and more. |
| [EXPEDITION_JOURNAL.md](EXPEDITION_JOURNAL.md) | Developers, LLM agents | First-hand account of building a 1,288-line production shortcut. Pipeline discovery, failure modes, workarounds, and strategic recommendations. |
| [SKILL.md](SKILL.md) | LLM agents (Claude Code) | Python API reference for `shortcuts_compiler.py`. Action catalog, control flow, builder helpers, composition patterns. Loaded as a Claude Code skill. |

## Research

Balanced Sashimi research docs live in [`research/docs/`](../research/docs/):

| Document | Description |
|----------|-------------|
| [BALANCED_SASHIMI_RESEARCH.md](../research/docs/BALANCED_SASHIMI_RESEARCH.md) | Theory: hybrid continuous-ternary architecture for domain-constrained program synthesis. |
| [BALANCED_SASHIMI_PLAN.md](../research/docs/BALANCED_SASHIMI_PLAN.md) | Operational plan: phases, gates, compute budget, stop/go criteria. |
| [EXPERIMENT_RESULTS.md](../research/docs/EXPERIMENT_RESULTS.md) | Results ledger: baselines, experiment data, metrics. |

See also [`research/README.md`](../research/README.md) for module structure and gate status.

## Archive

Historical analysis and development logs in [`docs/archive/`](archive/):

| Document | Description |
|----------|-------------|
| [CONTROL_FLOW_SUMMARY.md](archive/CONTROL_FLOW_SUMMARY.md) | Executive summary of control flow patterns across 303 corpus shortcuts. |
| [CONTROL_FLOW_REPORT.txt](archive/CONTROL_FLOW_REPORT.txt) | Detailed technical breakdown of the control flow study. |
| [README_CONTROL_FLOW_ANALYSIS.md](archive/README_CONTROL_FLOW_ANALYSIS.md) | Methodology and data dictionary for the corpus study. |
| [BLOCK_MISMATCH_ANALYSIS.md](archive/BLOCK_MISMATCH_ANALYSIS.md) | Deep dive on one structural anomaly found in the corpus. |
| [PHASE3_LOG.md](archive/PHASE3_LOG.md) | Implementation log for Phase 3: model training, Outlines grammar, linter development. |

## Key Files Elsewhere

| File | Purpose |
|------|---------|
| [`CLAUDE.md`](../CLAUDE.md) | Claude Code project instructions: commands, conventions, module map. |
| [`references/action_catalog.json`](../references/action_catalog.json) | 615-action catalog (ground truth for action names and parameters). |
| [`references/shortcutdsl.lark`](../references/shortcutdsl.lark) | DSL grammar (Lark LALR). |
| [`references/scenario_packs/`](../references/scenario_packs/) | 8 benchmark scenarios with rubrics and reference DSL. |
