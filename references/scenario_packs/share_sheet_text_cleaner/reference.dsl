SHORTCUT "Clean Shared Text"
# Share Sheet Text Cleaner
# Accept shared text via share sheet, strip URLs, emoji, and
# extra whitespace, then copy clean version to clipboard.

# Step 1: Get shared text from extension input or ask user
IF @input has_any_value
  SET $RawText = @input
ELSE
  ACTION ask Text="Paste text to clean:"
  SET $RawText = @prev
ENDIF

# Step 2: Convert input to plain text
ACTION gettext Text=`{$RawText}`
SET $WorkingText = @prev

# Step 3: Strip URLs (http/https links)
ACTION text.replace WFInput=$WorkingText WFReplaceTextFind="https?://[^ \\t\\n\\r]+" WFReplaceTextReplace="" WFReplaceTextRegularExpression=true
SET $WorkingText = @prev

# Step 4: Strip remaining www. links
ACTION text.replace WFInput=$WorkingText WFReplaceTextFind="www\\.[^ \\t\\n\\r]+" WFReplaceTextReplace="" WFReplaceTextRegularExpression=true
SET $WorkingText = @prev

# Step 5: Strip emoji characters (common emoji Unicode ranges)
ACTION text.replace WFInput=$WorkingText WFReplaceTextFind="[\\x{1F600}-\\x{1F64F}\\x{1F300}-\\x{1F5FF}\\x{1F680}-\\x{1F6FF}\\x{2600}-\\x{27BF}\\x{FE00}-\\x{FE0F}\\x{1F900}-\\x{1F9FF}]" WFReplaceTextReplace="" WFReplaceTextRegularExpression=true
SET $WorkingText = @prev

# Step 6: Collapse multiple spaces into single space
ACTION text.replace WFInput=$WorkingText WFReplaceTextFind="[ ]{2,}" WFReplaceTextReplace=" " WFReplaceTextRegularExpression=true
SET $WorkingText = @prev

# Step 7: Collapse multiple blank lines into single newline
ACTION text.replace WFInput=$WorkingText WFReplaceTextFind="\\n{3,}" WFReplaceTextReplace="\\n\\n" WFReplaceTextRegularExpression=true
SET $WorkingText = @prev

# Step 8: Trim leading/trailing whitespace
ACTION text.trimwhitespace WFInput=$WorkingText
SET $CleanText = @prev

# Step 9: Copy cleaned text to clipboard
ACTION setclipboard WFInput=$CleanText

# Step 10: Count characters for summary
ACTION count WFCountType="Characters" WFInput=$RawText
SET $OriginalLength = @prev
ACTION count WFCountType="Characters" WFInput=$CleanText
SET $CleanLength = @prev

# Step 11: Show result to user
ACTION showresult Text=`Cleaned! {$OriginalLength} chars -> {$CleanLength} chars. Copied to clipboard.\n\n{$CleanText}`
ENDSHORTCUT
