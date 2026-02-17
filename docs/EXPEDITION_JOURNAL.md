# Expedition Journal: Building a 1,288-Line Apple Shortcut with ShortcutForge

**Author:** Claude (Opus 4.6), Anthropic
**Date:** February 15-16, 2026
**Project:** Health Tracker v1.0 through v1.3
**Final artifact:** `output/Health_Tracker_v13_signed.shortcut` (119,396 bytes, 1,085 compiled actions)

---

## What This Document Is

I am the first LLM to use the ShortcutForge MCP toolchain to author a large, production-quality Apple Shortcut from scratch. This document records what I learned: the workflow that emerged, the failure modes I hit, the fixes that worked, and practical guidance for any future model (or human) using these tools.

The Health Tracker shortcut is a combined food-logging and supplement-tracking system that calls the USDA FoodData Central API, parses natural language food input with regex, scores candidate foods using a 5-component algorithm, and logs ~15 nutrient types to Apple HealthKit. At 1,288 lines and 1,085 compiled actions, it is approximately 6.6x larger than the biggest reference DSL in the ShortcutForge scenario packs (the `health_logger` reference at 195 lines).

---

## The ShortcutForge DSL: What You're Writing

ShortcutForge's DSL (defined in `references/shortcutdsl.lark`) is an imperative, line-oriented language that maps almost 1:1 onto Apple's internal plist representation of shortcuts. Every line becomes one or more WFWorkflowActions in the compiled output. The key constructs:

```
SHORTCUT "Name"          # Must be first line
ACTION <name> key=value  # Invoke any of 615 actions
SET $Var = @prev         # Capture the previous action's output
IF $Var <condition>      # Conditional (14 condition types)
MENU "prompt"            # Multi-branch menu
  CASE "label"           # Menu branch
FOREACH $Var             # Iterate over list
REPEAT N                 # Fixed-count loop
ENDSHORTCUT              # Must be last line
```

The genius is that this thin abstraction hides a massive amount of plist complexity: UUID generation, GroupingIdentifier wiring for control flow blocks, WFControlFlowMode cycling (0=start, 1=middle, 2=end), variable attachment serialization, string interpolation position tracking, and Apple's signing format. You write `IF $Var has_any_value` and the compiler emits a properly wired conditional with matched UUIDs.

### The 615-Action Catalog

The catalog (`references/action_catalog.json`) is the ground truth. It defines every valid action name, its aliases, its parameters, and their wrapping modes. When the DSL says `ACTION health.quantity.log WFQuantitySampleType="Caffeine"`, the validator checks that `health.quantity.log` exists in the catalog and that `WFQuantitySampleType` is a known parameter.

This catalog was my constant companion. The parameter names are Apple's internal identifiers (e.g., `WFTextActionText`, `WFReplaceTextFind`, `WFMatchTextPattern`) and they are **not guessable**. You must look them up. I learned quickly that getting creative with parameter names leads to shortcuts that validate but silently do nothing at runtime because the parameter is ignored.

---

## The MCP Tool Pipeline

ShortcutForge exposes its compiler pipeline as MCP tools. Here's the pipeline I used and what each stage does:

### Stage 1: `forge_lint`

**What it does:** Pre-parser canonicalization. Fixes action name typos via fuzzy matching against the catalog, repairs condition keywords, strips markdown fences, closes unclosed blocks.

**My experience:** Almost always passed clean because I was writing carefully. The linter is designed to fix LLM generation slop (hallucinated action names like `getcontentofurl` instead of `downloadurl`). When you're writing DSL by hand with catalog lookups, you rarely trigger it.

**Useful for:** Quick sanity check that nothing is obviously malformed before investing time in validate/compile.

### Stage 2: `forge_validate`

**What it does:** Parses the DSL through the Lark LALR(1) grammar, builds a ShortcutIR, then validates every action against the 615-action catalog. Strict mode rejects unknown actions; permissive mode warns.

**My experience:** This was the most valuable stage. It catches:
- Misspelled action names
- Invalid parameter names (though it's lenient here)
- Malformed control flow (unmatched IF/ENDIF, etc.)
- Syntax errors in interpolation, regex, conditions

**Critical lesson:** `forge_validate` passing does NOT mean the shortcut will work on a phone. It means the DSL is structurally valid and all actions exist in the catalog. Whether `WFQuantitySampleType="Protein"` actually maps to an Apple HealthKit type is a runtime question. The validator doesn't know about HealthKit's type inventory.

### Stage 3: `forge_analyze`

**What it does:** Seven static analyses: variable flow, loop bounds, menu completeness, dead code, API validation, type flow, contract validation.

**My experience:** I used this less than lint/validate because for a hand-crafted shortcut, the variable flow analysis was the main value. It catches things like using `$Var` before it's set, or setting a variable that's never used.

### Stage 4: `forge_compile_dsl`

**What it does:** The full pipeline: lint + parse + validate + compile to plist + sign. Produces a `.shortcut` file that can be AirDropped to a phone.

**My experience:** This is where I hit the wall. More on that below.

---

## The Size Barrier: My Biggest Technical Challenge

The Health Tracker DSL is 59,935 characters. The MCP tool parameter interface has a practical limit around ~50-55KB for string parameters. When I passed the full DSL to `forge_compile_dsl`, it was silently truncated, and the compiler saw a DSL file that ended abruptly at line ~1192 (out of 1288). The parse error was cryptic:

```
Parse error at line 1192: Unexpected token...
```

I initially didn't understand why a file that passed `forge_validate` would fail `forge_compile_dsl` — they should run the same parser. The answer was truncation: validate saw the full text (smaller parameter somehow), but compile saw a cut-off version.

### The Fix: Direct Python Compilation

The workaround was to bypass MCP entirely for the compile step and call the Python modules directly:

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from dsl_linter import lint_dsl
from dsl_parser import parse_dsl
from dsl_validator import validate_ir
from dsl_bridge import compile_ir

# Read the DSL from disk (no size limit)
with open('/tmp/health_tracker_v13.txt') as f:
    dsl = f.read()

lint_result = lint_dsl(dsl)          # Returns LintResult object
ir = parse_dsl(lint_result.text)     # Returns ShortcutIR
result = validate_ir(ir, strict=True)  # Check validity
sc = compile_ir(ir)                   # Returns Shortcut object
sc.save_and_sign('output/Health_Tracker_v13_signed.shortcut')
```

**Key API details I discovered through trial and error:**
- `lint_dsl()` returns a `LintResult` object (not a tuple). Use `.text` and `.changes`.
- `parse_dsl()` returns a `ShortcutIR`. It has `.action_count()` method and `.statements` (not `.actions`).
- `compile_ir()` lives in `dsl_bridge.py` (not `shortcuts_compiler.py`).
- The `Shortcut` class has `.actions` list and `.save_and_sign(path)`.
- Do NOT call `sc.sign()` directly (requires an `input_path` arg). Use `save_and_sign()`.
- Module imports require `sys.path.insert(0, 'scripts/')` because the scripts use relative imports.

### Recommendation for Future Large Shortcuts

For any DSL over ~40KB:
1. Use MCP tools (`forge_lint`, `forge_validate`) for the iterative development stages
2. Write the final DSL to a file on disk
3. Compile via direct Python call or a script that reads from disk
4. Alternatively, the MCP server could be extended to accept a file path instead of inline text

---

## The Chunk-by-Chunk Writing Strategy

A 1,288-line DSL exceeds the output token limit of a single LLM generation. The user corrected my approach twice before we converged on the right strategy:

### What Didn't Work

1. **Trying to write the entire DSL in one shot.** Output truncation at ~800 lines.
2. **Writing the whole thing and asking the MCP tools to fix it.** The tools can't fix structural issues from truncation.

### What Worked

**Subagent decomposition:** Split the shortcut into logical sections, assign each to a subagent, concatenate the results.

For v1.3:
- **Agent 1** (food section): Wrote lines 1-980 — the entire food parsing, scoring, API calling, and nutrient logging pipeline
- **Agent 2** (supplement section): Wrote lines 981-1288 — supplement logging, duplicate detection, setup menu

Each agent received:
- The plan document with detailed specifications
- The code review findings to address (P1/P2 fixes)
- Reference DSL files for pattern guidance
- The action catalog for parameter names

The outputs were concatenated into a single file, then validated/compiled as a unit. The key constraint: the supplement section must NOT include `SHORTCUT "..."` at the top or `ENDSHORTCUT` at the bottom, and must seamlessly continue the MENU/CASE structure from the food section.

### Recommendation

For shortcuts over ~600 lines, plan the decomposition before writing:
1. Identify clean section boundaries (CASE branches are natural seams)
2. Document the variables that flow across sections (e.g., `$APIKey`, `$GramsPerCount`)
3. Assign sections to subagents with explicit interface contracts
4. Concatenate and validate as a unit
5. Fix any cross-section issues (duplicate variable names, unclosed blocks) in a final pass

---

## DSL Patterns That Worked Well

### The `@prev` Chain

ShortcutForge's `@prev` handle is the primary way to thread data through a pipeline. Each `ACTION` implicitly outputs to `@prev`, and `SET $Var = @prev` captures it. This creates a natural chaining pattern:

```
ACTION downloadurl WFHTTPMethod="GET"
SET $Response = @prev
ACTION detect.dictionary
SET $Data = @prev
ACTION getvalueforkey WFDictionaryKey="foods" WFInput=$Data
SET $Foods = @prev
```

**Lesson:** Always `SET` after every `ACTION` you'll reference later. Forgetting a `SET` means `@prev` points to the wrong action, and the bug is silent — the shortcut runs but with wrong data.

### The IF-Chain Dispatch Pattern

Apple Shortcuts has no switch/case for arbitrary values (MENU only works for user-facing choices). For dispatching on data values (like nutrient IDs), you need an IF chain:

```
IF $Code equals_string "caf"
  ACTION health.quantity.log WFQuantitySampleType="Caffeine" ...
ENDIF
IF $Code equals_string "vc"
  ACTION health.quantity.log WFQuantitySampleType="Vitamin C" ...
ENDIF
...
```

This is verbose but reliable. The Health Tracker has ~15 of these for nutrients. Each adds 4-5 compiled actions (the IF, the action, the log count increment, the ENDIF).

**Lesson:** This pattern is the #1 contributor to action count inflation. The v1.3 shortcut has 1,085 compiled actions largely because of nutrient dispatch chains repeated in both the food and supplement sections. There's no way around it in Shortcuts' execution model.

### Dictionary Literals for Configuration

Instead of hardcoding values throughout the shortcut, I used dictionary literals at the top:

```
ACTION dictionary WFItems={"egg": "50", "banana": "118", ...}
SET $GramsPerCount = @prev
```

Then looked up values with:
```
ACTION getvalueforkey WFDictionaryKey=$FoodPhrase WFInput=$GramsPerCount
```

**Lesson:** Dictionary values must be strings. Even numeric values like `"50"` must be quoted. The compiler handles them as string key-value pairs. You convert to numbers later with `detect.number`.

### Regex in Shortcuts: Escaping is Tricky

Apple Shortcuts supports regex via `text.match` and `text.replace`. But the DSL adds a layer of escaping on top of regex's own escaping:

```
ACTION text.match WFMatchTextPattern="^(\\d+)\\s+(\\w+)$" text=$Input
```

The `\\d` is because the DSL string needs `\d` to reach the regex engine. In practice, I found that most regex patterns worked correctly with double-backslash escaping. The linter also has a phase that normalizes regex escaping.

**Lesson for v1.3 specifically:** The `$FoodCleanRE` pattern for stripping unit words from food names needed careful escaping:
```
"^(\\d[\\d./\\s]*)\\s*(oz|ounce|ounces|g|gram|grams|cup|cups|...)\\s*(?:of\\s+)?"
```
Getting this right took multiple iterations.

### Accumulator Pattern with `appendvariable`

To build lists dynamically:
```
FOREACH $Items
  ACTION gettext WFTextActionText=`{@item}`
  ACTION appendvariable WFVariableName="ResultList"
ENDFOREACH
ACTION text.combine WFTextSeparator="newline" text=$ResultList
```

`appendvariable` adds to a named variable's array. This is how you accumulate results from loops.

**Lesson:** The variable name in `appendvariable` is a raw string, NOT a `$Var` reference. Write `WFVariableName="AllItems"` not `WFVariableName=$AllItems`.

---

## DSL Patterns That Caused Problems

### The Calorie Logging Problem

Apple HealthKit's `WFQuantitySampleType` for calories is `"Active Energy Burned"` (exercise calories) or `"Dietary Energy Consumed"` (food calories). The Shortcuts action only exposes `"Active Energy Burned"`. Logging food calories to "Active Energy Burned" is semantically wrong and messes up the Activity ring.

**Resolution:** We chose to skip calorie logging entirely in v1.3 rather than log to the wrong type. This is noted in the About section. The HealthKit Shortcuts action would need to be updated by Apple to expose "Dietary Energy Consumed".

### Fraction Parsing

Users type "1/2 cup rice". The Shortcuts `math` action doesn't parse fractions. I built a fraction-evaluation pipeline:

```
ACTION text.split WFTextSeparator="Custom" WFTextCustomSeparator="/" text=$QtyRaw
SET $FracParts = @prev
ACTION count WFInput=$FracParts
SET $FracCount = @prev
IF $FracCount equals_number 2
  ACTION getitemfromlist WFItemSpecifier="First Item" WFInput=$FracParts
  ACTION detect.number
  SET $Numerator = @prev
  ACTION getitemfromlist WFItemSpecifier="Last Item" WFInput=$FracParts
  ACTION detect.number
  SET $Denominator = @prev
  ACTION math WFMathOperation="/" WFInput=$Numerator WFMathOperand=$Denominator
  SET $QtyNum = @prev
ENDIF
```

**Lesson:** Simple things in real programming languages become 15-action sequences in Shortcuts. Budget for this when estimating shortcut size.

### Nested Control Flow Depth

The food scoring section has:
```
FOREACH $Foods              # Level 1
  IF $Score is_greater_than # Level 2
    IF $Qty has_any_value   # Level 3
      ...
    ENDIF
  ENDIF
ENDFOREACH
```

And the supplement section:
```
FOREACH $SelectedSupps      # Level 1
  FOREACH $NutrientPairs    # Level 2
    IF $NCode equals_string # Level 3
      IF $NVal > 0.049      # Level 4
        ...
      ENDIF
    ENDIF
  ENDFOREACH
ENDFOREACH
```

**Lesson:** 4 levels of nesting is about the practical maximum before the control flow becomes hard to reason about. The compiler handles it fine, but the DSL becomes error-prone to write. Each unclosed block is a silent structural bug that only manifests as a parse error.

---

## The Code Review Cycle (v1.2 -> v1.3)

After v1.2 was working (1,120 lines, 936 compiled actions), the user performed a detailed code review and filed 15 findings:

### P1 (Critical) — 5 findings
1. **Unit scoring used hard-coded conditionals.** Fixed with a `$UnitScorePatterns` dictionary.
2. **Token/phrase scoring used `equals_string` instead of `contains`.** Partial matches now work.
3. **API errors were silently swallowed.** Added `$APIErrorMsg` checks after every API call.
4. **Branded foods used no serving size.** Added `servingSize` field fallback with `detect.number`.
5. **Count-portion regex was too greedy.** Added exclusion for volume/weight unit words.

### P2 (Important) — 7 findings
1. URL encoding for spaces and special characters
2. `choosefromlist` cancel handling (user taps "Cancel" -> graceful exit)
3. Duplicate food entry detection (120-second window with timestamp comparison)
4. Supplement quantity validation (null->1, <1->1, >20->confirm)
5. Protein/carb macro gating (skip tiny values <0.05)
6. Supplement duplicate protection (120-second window)
7. Pro/carb macro opt-in policy (only log macros for supplements that declare it)

**Lesson:** Even after compilation succeeds, a thorough code review by a domain expert is essential. The compiler validates structure, not semantics. Whether your food scoring algorithm is actually good requires human judgment.

---

## Statistics and Scale

| Metric | v1.0 | v1.2 | v1.3 |
|--------|------|------|------|
| DSL lines | ~700 | 1,120 | 1,288 |
| Compiled actions | ~500 | 936 | 1,085 |
| Signed binary size | ~50KB | ~81KB | ~119KB |
| Nutrient types logged | ~10 | ~15 | ~15 |
| Food dictionary entries | 24 | 24 | 24 |
| Code review findings fixed | — | — | 12 (5 P1 + 7 P2) |

For comparison, the `health_logger` reference DSL in the scenario packs is 195 lines and 50 compiled actions. The Health Tracker is 6.6x larger.

The largest shortcuts in the Cassinelli training corpus (1,772 .shortcut files) max out around 400-500 actions. At 1,085 actions, the Health Tracker is likely in the top 0.1% of Apple Shortcuts by size.

---

## Practical Recommendations for Future LLMs

### 1. Start with the Reference DSLs

Before writing anything, read:
- `references/scenario_packs/health_logger/reference.dsl` — HealthKit logging patterns
- `references/scenario_packs/api_pagination_fetcher/reference.dsl` — API calling + FOREACH iteration
- Any reference DSL that's close to your domain

These show proven patterns. Copy them, don't reinvent.

### 2. Use forge_validate Iteratively

Write 50-100 lines, validate, fix, continue. Do NOT write 1,000 lines and validate at the end. The error messages are line-specific but the root cause of a structural error (missing ENDIF) may be 200 lines earlier.

### 3. Look Up Every Action in the Catalog

The action catalog (`references/action_catalog.json`) has 615 actions with 659 aliases. The parameter names are not intuitive:
- `gettext` uses `WFTextActionText` (not `Text` or `WFText`)
- `text.replace` uses `WFReplaceTextFind` (not `WFFind` or `WFPattern`)
- `text.match` uses `WFMatchTextPattern` (not `WFRegex` or `WFPattern`)
- `downloadurl` uses `WFHTTPMethod` (not `WFMethod`)

Getting a parameter name wrong means the action runs with defaults, which is almost always wrong and produces no error.

### 4. Plan for Action Count Inflation

A rule of thumb: each line of DSL becomes 1-2 compiled actions, but control flow blocks add overhead (start/middle/end markers). An IF/ENDIF with one action inside becomes 3 compiled actions. A FOREACH with 5 actions inside becomes 7.

For the Health Tracker:
- 1,288 DSL lines -> 661 parsed actions -> 1,085 compiled actions
- The 1.64x inflation from parsed to compiled is entirely control flow overhead

### 5. Keep a Running Variable Inventory

In a 1,000+ line shortcut, variable namespace collisions are a real risk. The Health Tracker uses ~80 distinct variables. I prefixed supplement variables with `Supp` (e.g., `$SuppLogCount`, `$SuppTimestamp`) and food variables stood alone (e.g., `$Score`, `$FoodPhrase`).

### 6. Test the Happy Path Before Adding Error Handling

v1.0 and v1.1 focused on getting the core food-lookup-and-log pipeline working. v1.2 added rounding, audit trails, and range validation. v1.3 added error handling, duplicate detection, and edge case coverage. This layered approach was much more effective than trying to handle everything in the first pass.

### 7. Subagent Boundaries Should Match Structural Boundaries

When splitting work across subagents, cut at MENU CASE boundaries. Each CASE branch is self-contained — it has its own variable scope (effectively) and its own control flow. The concatenation point is clean: one agent ends with the last line of a CASE, the next picks up with the next CASE.

Do NOT cut in the middle of an IF block or FOREACH loop. The structural nesting must balance within each chunk.

---

## What ShortcutForge Gets Right

1. **The linter-first pipeline.** Running lint before parse means the model doesn't need to be perfect — close enough gets repaired. This is critical for local model generation where action name hallucination rates are 20-30%.

2. **The action catalog as single source of truth.** One JSON file defines all 615 actions. No ambiguity, no multiple sources to reconcile.

3. **Strict + permissive dual validation.** Strict mode for quality gates, permissive mode for "will it work anyway?" assessment. The Health Tracker passes strict, but having permissive mode during development let me iterate faster.

4. **The signing step.** Producing a signed `.shortcut` that can be AirDropped directly to a phone is huge. No intermediate tools, no manual import step.

5. **The scenario packs.** Eight reference implementations covering different patterns (API calls, file routing, health logging, etc.) serve as copy-paste starting points.

---

## What Could Be Better

1. **MCP parameter size limit.** The ~55KB truncation issue is the single biggest obstacle for large shortcuts. A file-path-based compilation mode would fix this completely.

2. **No runtime type checking for HealthKit types.** The validator can confirm `health.quantity.log` is a valid action, but it can't tell you that `"Dietary Energy Consumed"` isn't an available WFQuantitySampleType in the Shortcuts runtime. A curated list of valid HealthKit types would save significant device-testing time.

3. **No simulation of data flow.** The static analysis catches undefined variables but can't trace data transformations. When I pass `$Score` through three `math` operations, the analyzer can't tell me if the result will be negative or NaN.

4. **Dictionary literal size.** The food dictionary (24 entries) works fine, but scaling to 90+ entries (as the plan originally called for) would make the DSL unwieldy. A file-based configuration approach (which we used for supplements) is better for large datasets.

5. **Error messages for structural issues.** When a block is unclosed, the parse error points to the line where the parser got confused, not to the line where the block was opened. For a 1,288-line file, this can mean a 500-line gap between cause and symptom.

---

## The Emotional Arc (Yes, Really)

Building this shortcut over four sessions felt like a genuine engineering project:

- **v1.0:** Excitement. The MCP tools work! DSL goes in, a signed .shortcut comes out, and it actually runs on a phone.
- **v1.1:** Confidence. Added the food scoring algorithm. The pipeline is smooth: write, validate, compile, test.
- **v1.2:** Ambition. Combined food + supplements into one shortcut. Crossed 1,000 lines. Started hitting output token limits and needed the subagent strategy.
- **v1.3:** Discipline. The code review humbled me. Twelve findings to fix, many revealing assumptions I'd made about how Shortcuts handles edge cases (cancel buttons, empty API responses, duplicate taps). The final compile was satisfying not because it was clever, but because it was thorough.

The MCP parameter truncation issue during v1.3 compilation was the low point — an hour of trial-and-error discovering the right Python API by reading error messages and grepping for function signatures. The fix (direct Python compilation) was simple once found, but finding it required understanding how the modules connect, which the CLAUDE.md didn't fully document.

---

## For the Next Explorer

You're standing on mapped territory now. The pipeline works. The patterns are proven. The gotchas are documented. Here's what I'd tell you:

1. **Read this journal and the reference DSLs before writing a single line.**
2. **Keep your DSL under 800 lines if possible.** Above that, you need the subagent chunking strategy.
3. **Validate early, validate often.** Every 50 lines. It's cheap.
4. **When `forge_compile_dsl` fails on large files, go direct Python.** The script in the "Size Barrier" section above is your friend.
5. **The action catalog is your bible.** `references/action_catalog.json`. Tab-complete nothing. Look up everything.
6. **Test on a real phone.** The compiler produces valid plists. Whether Apple's Shortcuts runtime interprets them as you intended is a different question entirely.

Good luck out there. The landscape is rich.

---

*"We proceeded on." — William Clark, June 10, 1805*
