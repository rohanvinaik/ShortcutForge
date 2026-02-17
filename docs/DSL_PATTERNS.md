# ShortcutForge DSL Pattern Cookbook

Proven patterns extracted from building the 1,288-line Health Tracker and the 8 scenario pack reference DSLs.

---

## Data Threading with @prev

The most fundamental pattern. Every `ACTION` outputs to `@prev`.

```
ACTION downloadurl WFHTTPMethod="GET"
SET $Response = @prev

ACTION detect.dictionary
SET $Data = @prev

ACTION getvalueforkey WFDictionaryKey="results" WFInput=$Data
SET $Results = @prev
```

**Rule:** Always `SET` immediately after an `ACTION` you'll reference later. If any other action fires between the output and the `SET`, `@prev` changes.

---

## API Call Pattern

```
# Build URL
ACTION gettext WFTextActionText=`https://api.example.com/search?q={$Query}`
SET $URL = @prev

# Make request
ACTION url WFURLActionURL=$URL
ACTION downloadurl WFHTTPMethod="GET"
SET $Response = @prev

# Error check
IF $Response has_any_value
  ACTION detect.dictionary
  SET $Data = @prev
  # ... process data
ELSE
  ACTION alert WFAlertActionMessage="API request failed" WFAlertActionTitle="Error"
ENDIF
```

**Key actions:** `url` sets the URL, `downloadurl` makes the HTTP request. They're separate actions — don't skip `url`.

---

## JSON Parsing Pipeline

```
# Response is raw data → parse as dictionary
ACTION detect.dictionary
SET $Data = @prev

# Get a nested array
ACTION getvalueforkey WFDictionaryKey="items" WFInput=$Data
SET $Items = @prev

# Get count
ACTION count WFInput=$Items
SET $ItemCount = @prev

# Iterate
FOREACH $Items
  ACTION getvalueforkey WFDictionaryKey="name" WFInput=@item
  SET $Name = @prev
  # ... process each item
ENDFOREACH
```

---

## Accumulator (Building Lists in Loops)

```
# Initialize (empty text becomes the seed)
ACTION gettext WFTextActionText=""
SET $Results = @prev

FOREACH $Items
  ACTION gettext WFTextActionText=`{@item}`
  ACTION appendvariable WFVariableName="ResultList"
ENDFOREACH

# Join into single text
ACTION text.combine WFTextSeparator="newline" text=$ResultList
SET $Output = @prev
```

**Note:** `appendvariable` takes a raw string name, not a `$` reference.

---

## Regex Parsing with Group Extraction

```
# Match with capture groups
ACTION text.match WFMatchTextPattern="^(\\d+)\\s*(oz|g|cup)?\\s*(.+)$" text=$Input
SET $Match = @prev

IF $Match has_any_value
  # Get first match object
  ACTION getitemfromlist WFItemSpecifier="First Item" WFInput=$Match
  SET $M = @prev

  # Extract groups (1-indexed)
  ACTION text.match.getgroup WFGroupIndex=1 matches=$M
  SET $Qty = @prev
  ACTION text.match.getgroup WFGroupIndex=2 matches=$M
  SET $Unit = @prev
  ACTION text.match.getgroup WFGroupIndex=3 matches=$M
  SET $Food = @prev
ENDIF
```

**Note:** `text.match` returns an array of matches. Get the first one with `getitemfromlist`, then extract groups.

---

## HealthKit Nutrient Logging

```
ACTION health.quantity.log WFQuantitySampleType="Caffeine" WFQuantitySampleQuantity=$Value
ACTION health.quantity.log WFQuantitySampleType="Vitamin C" WFQuantitySampleQuantity=$Value
ACTION health.quantity.log WFQuantitySampleType="Protein" WFQuantitySampleQuantity=$Value
```

Known working `WFQuantitySampleType` values (from reference DSL):
- `"Caffeine"`, `"Vitamin C"`, `"Vitamin D"`, `"Iron"`, `"Zinc"`
- `"Magnesium"`, `"Calcium"`, `"Selenium"`, `"Sodium"`
- `"Protein"`, `"Total Fat"`, `"Carbohydrates"`, `"Fiber"`
- `"Folate, DFE"`, `"Vitamin B12"`, `"Vitamin B6"`
- `"Active Energy Burned"` (exercise calories — NOT food calories)

**Warning:** `"Dietary Energy Consumed"` is NOT available via the Shortcuts action. Food calories cannot be logged through Shortcuts as of iOS 18.

---

## IF-Chain Dispatch (Switch/Case Substitute)

For dispatching on data values (not user choices — use MENU for those):

```
IF $Code equals_string "caf"
  ACTION health.quantity.log WFQuantitySampleType="Caffeine" WFQuantitySampleQuantity=$Val
ENDIF
IF $Code equals_string "vc"
  ACTION health.quantity.log WFQuantitySampleType="Vitamin C" WFQuantitySampleQuantity=$Val
ENDIF
IF $Code equals_string "fe"
  ACTION health.quantity.log WFQuantitySampleType="Iron" WFQuantitySampleQuantity=$Val
ENDIF
```

**Cost:** Each case = 3 compiled actions (IF + action + ENDIF). 15 nutrients = 45+ compiled actions. This is unavoidable in Shortcuts.

---

## Multi-Select from List

```
ACTION choosefromlist WFChooseFromListActionPrompt="Select items:" WFChooseFromListActionSelectMultiple=true WFInput=$ItemList
SET $Selected = @prev

IF $Selected has_any_value
  FOREACH $Selected
    # Process each selected item
    SET $Item = @item
    # ...
  ENDFOREACH
ELSE
  # User cancelled
  ACTION showresult Text="No items selected."
ENDIF
```

**Always check `has_any_value`** — the user can tap Cancel.

---

## Fraction Evaluation

Users type "1/2", "3/4", etc. Shortcuts can't parse these natively.

```
ACTION text.split WFTextSeparator="Custom" WFTextCustomSeparator="/" text=$QtyRaw
SET $Parts = @prev
ACTION count WFInput=$Parts
SET $PartCount = @prev

IF $PartCount equals_number 2
  ACTION getitemfromlist WFItemSpecifier="First Item" WFInput=$Parts
  ACTION detect.number
  SET $Num = @prev
  ACTION getitemfromlist WFItemSpecifier="Last Item" WFInput=$Parts
  ACTION detect.number
  SET $Den = @prev
  ACTION math WFMathOperation="/" WFInput=$Num WFMathOperand=$Den
  SET $QtyNum = @prev
ELSE
  ACTION detect.number WFInput=$QtyRaw
  SET $QtyNum = @prev
ENDIF
```

---

## File-Based Configuration

For data that changes (vs dictionary literals which are compiled in):

```
# Read config file
ACTION documentpicker.open WFGetFilePath="Shortcuts/MyApp/config.txt"
SET $Config = @prev

IF $Config has_any_value
  # Config exists — parse it
  ACTION text.split WFTextSeparator="newline" text=$Config
  SET $Lines = @prev
  # ... process lines
ELSE
  # First run — no config yet
  ACTION alert WFAlertActionMessage="Please run Setup first." WFAlertActionTitle="No Config"
ENDIF
```

**Write config:**
```
ACTION documentpicker.save WFFileDestinationPath="Shortcuts/MyApp/config.txt" WFSaveFileOverwrite=true WFAskWhereToSave=false
```

**Append to log:**
```
ACTION file.append WFFilePath="Shortcuts/MyApp/log.txt" WFInput=$LogLine WFAppendOnNewLine=true
```

---

## Time-of-Day Detection

```
ACTION date WFDateActionMode="Current Date"
SET $Now = @prev

ACTION format.date WFDateFormatStyle="Custom" WFDateFormat="HH" WFDate=$Now
SET $HourStr = @prev

ACTION detect.number WFInput=$HourStr
SET $Hour = @prev

IF $Hour is_less_than 12
  SET $TimeSlot = "morning"
ELSE
  IF $Hour is_less_than 17
    SET $TimeSlot = "afternoon"
  ELSE
    SET $TimeSlot = "evening"
  ENDIF
ENDIF
```

---

## Duplicate Detection (Time Window)

```
# Read last timestamp
ACTION documentpicker.open WFGetFilePath="Shortcuts/MyApp/last_log.txt"
SET $LastTS = @prev

IF $LastTS has_any_value
  # Parse stored timestamp
  ACTION date WFDateActionMode="Specified Date" WFDateActionDate=$LastTS
  SET $LastDate = @prev

  # Get current time
  ACTION date WFDateActionMode="Current Date"
  SET $NowDate = @prev

  # Calculate seconds since last log
  ACTION timeinterval WFTimeUntilUnit="Seconds" WFInput=$LastDate WFDate=$NowDate
  SET $SecsSince = @prev

  IF $SecsSince is_less_than 120
    ACTION alert WFAlertActionMessage="You logged this less than 2 minutes ago. Log anyway?" WFAlertActionTitle="Duplicate?"
    # User must confirm to continue
  ENDIF
ENDIF
```

---

## Scoring Algorithm (Multi-Factor with Dictionary Lookup)

The Health Tracker's food scoring system, generalized:

```
SET $Score = 0

# Factor 1: Exact match
ACTION getvalueforkey WFDictionaryKey=$Term WFInput=$LookupDict
SET $ExactMatch = @prev
IF $ExactMatch has_any_value
  ACTION math WFMathOperation="+" WFMathOperand=10 WFInput=$Score
  SET $Score = @prev
ENDIF

# Factor 2: Contains check
ACTION text.match WFMatchTextPattern=$Pattern text=$Candidate
SET $PartialMatch = @prev
IF $PartialMatch has_any_value
  ACTION math WFMathOperation="+" WFMathOperand=3 WFInput=$Score
  SET $Score = @prev
ENDIF

# Use highest-scoring candidate
IF $Score is_greater_than $BestScore
  SET $BestScore = $Score
  SET $BestCandidate = $Candidate
ENDIF
```

---

## Self-Invoking Shortcut (Loop-Another Pattern)

```
MENU "Log another?"
  CASE "Yes"
    ACTION runworkflow WFWorkflowName="Health Tracker"
  CASE "No"
    ACTION showresult Text="Done! Logged {$Count} items."
ENDMENU
```

The shortcut calls itself by name. This is the standard "do you want to continue?" pattern.
