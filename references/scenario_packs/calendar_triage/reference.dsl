SHORTCUT "Calendar Triage"
# Calendar Triage
# Get today's calendar events, categorize them by calendar,
# build a grouped summary, and display the result.

# Step 1: Get today's date
ACTION date WFDateActionMode="Current Date"
SET $Today = @prev

# Step 2: Format today's date for display
ACTION format.date WFDateFormatStyle="Medium" WFDate=$Today
SET $TodayFormatted = @prev

# Step 3: Get upcoming events for today
ACTION getupcomingevents WFGetUpcomingItemCount=50
SET $AllEvents = @prev

# Step 4: Count total events
ACTION count WFInput=$AllEvents
SET $EventCount = @prev

# Step 5: Check if we have any events
IF $EventCount equals_number 0
  ACTION alert WFAlertActionMessage="No events found for today." WFAlertActionTitle="Calendar Triage"
ELSE
  # Step 6: Initialize category accumulators
  ACTION gettext Text=""
  SET $WorkEvents = @prev
  ACTION gettext Text=""
  SET $PersonalEvents = @prev
  ACTION gettext Text=""
  SET $OtherEvents = @prev
  SET $WorkCount = 0
  SET $PersonalCount = 0
  SET $OtherCount = 0

  # Step 7: Process each event
  FOREACH $AllEvents
    # Extract event title
    ACTION properties.calendarevents WFContentItemPropertyName="Title" WFInput=@item
    SET $EventTitle = @prev

    # Extract event start time
    ACTION properties.calendarevents WFContentItemPropertyName="Start Date" WFInput=@item
    SET $StartDate = @prev

    # Format the start time
    ACTION format.date WFDateFormatStyle="Short" WFTimeFormatStyle="Short" WFDate=$StartDate
    SET $StartFormatted = @prev

    # Extract calendar name
    ACTION properties.calendarevents WFContentItemPropertyName="Calendar" WFInput=@item
    SET $CalendarName = @prev

    # Build the event line
    ACTION gettext Text=`  {$StartFormatted} - {$EventTitle}`
    SET $EventLine = @prev

    # Categorize by calendar name
    IF $CalendarName contains "Work"
      ACTION appendvariable WFVariableName="WorkEvents"
      ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$WorkCount
      SET $WorkCount = @prev
    ELSE
      IF $CalendarName contains "Personal"
        ACTION appendvariable WFVariableName="PersonalEvents"
        ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$PersonalCount
        SET $PersonalCount = @prev
      ELSE
        ACTION appendvariable WFVariableName="OtherEvents"
        ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$OtherCount
        SET $OtherCount = @prev
      ENDIF
    ENDIF
  ENDFOREACH

  # Step 8: Build the grouped summary
  ACTION gettext Text=`Calendar Triage - {$TodayFormatted}\n{$EventCount} events total\n`
  SET $Summary = @prev

  # Add Work section if events exist
  IF $WorkCount is_greater_than 0
    ACTION text.combine WFTextSeparator="newline" WFInput=$WorkEvents
    SET $WorkList = @prev
    ACTION gettext Text=`{$Summary}\nWork ({$WorkCount}):\n{$WorkList}\n`
    SET $Summary = @prev
  ENDIF

  # Add Personal section if events exist
  IF $PersonalCount is_greater_than 0
    ACTION text.combine WFTextSeparator="newline" WFInput=$PersonalEvents
    SET $PersonalList = @prev
    ACTION gettext Text=`{$Summary}\nPersonal ({$PersonalCount}):\n{$PersonalList}\n`
    SET $Summary = @prev
  ENDIF

  # Add Other section if events exist
  IF $OtherCount is_greater_than 0
    ACTION text.combine WFTextSeparator="newline" WFInput=$OtherEvents
    SET $OtherList = @prev
    ACTION gettext Text=`{$Summary}\nOther ({$OtherCount}):\n{$OtherList}\n`
    SET $Summary = @prev
  ENDIF

  # Step 9: Display the final summary
  ACTION showresult Text=`{$Summary}`
ENDIF
ENDSHORTCUT
