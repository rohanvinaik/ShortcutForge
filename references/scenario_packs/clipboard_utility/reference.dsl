SHORTCUT "Clipboard Utility"
# Clipboard Utility
# Menu-driven tool: clean clipboard, append to running list, search history

MENU "Clipboard Utility"
CASE "Clean Clipboard"
  # Get current clipboard contents
  ACTION getclipboard
  SET $RawClip = @prev

  IF $RawClip has_any_value
    # Strip whitespace by using text.replace to remove leading/trailing spaces
    ACTION text.replace WFInput=$RawClip WFReplaceTextFind="^\\s+|\\s+$" WFReplaceTextReplace="" WFReplaceTextRegularExpression=true
    SET $Cleaned = @prev

    # Set the cleaned text back to clipboard
    ACTION setclipboard WFInput=$Cleaned

    ACTION showresult Text=`Clipboard cleaned successfully.`
  ELSE
    ACTION alert WFAlertActionMessage="Clipboard is empty. Nothing to clean." WFAlertActionTitle="No Content"
  ENDIF

CASE "Append to List"
  # Get current clipboard contents
  ACTION getclipboard
  SET $ClipContent = @prev

  IF $ClipContent has_any_value
    # Get current timestamp
    ACTION date WFDateActionMode="Current Date"
    SET $Now = @prev
    ACTION format.date WFDateFormatStyle="Short" WFInput=$Now
    SET $Timestamp = @prev

    # Build the entry to append
    ACTION gettext Text=`[{$Timestamp}] {$ClipContent}`
    SET $Entry = @prev

    # Append to the Clipboard History note
    ACTION appendnote WFInput=$Entry WFNote="Clipboard History"

    ACTION showresult Text=`Appended to Clipboard History note.`
  ELSE
    ACTION alert WFAlertActionMessage="Clipboard is empty. Nothing to append." WFAlertActionTitle="No Content"
  ENDIF

CASE "Search History"
  # Ask user for a search term
  ACTION ask Text="Enter search term:"
  SET $SearchTerm = @prev

  IF $SearchTerm has_any_value
    # Find notes matching the search term
    ACTION filter.notes WFContentItemFilter=$SearchTerm
    SET $Results = @prev

    IF $Results has_any_value
      ACTION showresult Text=`Found matching notes for: {$SearchTerm}`
    ELSE
      ACTION alert WFAlertActionMessage=`No clipboard history entries found matching: {$SearchTerm}` WFAlertActionTitle="No Results"
    ENDIF
  ELSE
    ACTION alert WFAlertActionMessage="No search term entered." WFAlertActionTitle="Cancelled"
  ENDIF

ENDMENU
ENDSHORTCUT
