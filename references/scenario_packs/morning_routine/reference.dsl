SHORTCUT "Morning Briefing"
# Morning Routine Briefing
# Time-aware routine: weather, calendar preview, commute time, spoken briefing

# Step 1: Get current date and time
ACTION date WFDateActionMode="Current Date"
SET $Now = @prev
ACTION format.date WFDateFormatStyle="Long" WFInput=$Now
SET $DateStr = @prev
ACTION format.date WFDateFormatStyle="Short" WFTimeFormatStyle="Short" WFInput=$Now
SET $TimeStr = @prev

# Step 2: Get current weather conditions
ACTION weather.currentconditions
SET $Weather = @prev

IF $Weather has_any_value
  ACTION getvalueforkey WFDictionaryKey="Temperature" WFInput=$Weather
  SET $Temperature = @prev
  ACTION getvalueforkey WFDictionaryKey="Condition" WFInput=$Weather
  SET $Condition = @prev
  ACTION getvalueforkey WFDictionaryKey="Humidity" WFInput=$Weather
  SET $Humidity = @prev
  ACTION gettext Text=`Weather: {$Temperature} and {$Condition}, humidity {$Humidity}.`
  SET $WeatherSummary = @prev
ELSE
  ACTION gettext Text="Weather: Unable to retrieve current conditions."
  SET $WeatherSummary = @prev
ENDIF

# Step 3: Get today's calendar events
ACTION filter.calendarevents WFContentItemFilter="Start Date" WFContentItemSortProperty="Start Date" WFContentItemSortOrder="Oldest First" WFContentItemLimitEnabled=true WFContentItemLimitNumber=10
SET $Events = @prev

IF $Events has_any_value
  # Count events
  ACTION count WFInput=$Events
  SET $EventCount = @prev

  # Build calendar summary
  ACTION gettext Text=`Calendar: {$EventCount} events today.`
  SET $CalendarSummary = @prev

  # Get details of the first event for commute calculation
  ACTION getvalueforkey WFDictionaryKey="0" WFInput=$Events
  SET $FirstEvent = @prev

  ACTION properties.calendarevents WFInput=$FirstEvent WFContentItemPropertyName="Title"
  SET $FirstTitle = @prev

  ACTION properties.calendarevents WFInput=$FirstEvent WFContentItemPropertyName="Start Date"
  SET $FirstStart = @prev

  ACTION properties.calendarevents WFInput=$FirstEvent WFContentItemPropertyName="Location"
  SET $FirstLocation = @prev

  # Build events list
  SET $EventsList = ""
  SET $EventIndex = 0
  FOREACH $Events
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$EventIndex
    SET $EventIndex = @prev

    ACTION properties.calendarevents WFInput=@item WFContentItemPropertyName="Title"
    SET $EvTitle = @prev

    ACTION properties.calendarevents WFInput=@item WFContentItemPropertyName="Start Date"
    SET $EvStart = @prev

    ACTION gettext Text=`{$EventsList}\n  {$EventIndex}. {$EvTitle} at {$EvStart}`
    SET $EventsList = @prev
  ENDFOREACH

  # Step 4: Calculate commute time to first meeting
  IF $FirstLocation has_any_value
    ACTION gettraveltime WFDestination=$FirstLocation WFGetDirectionsActionMode="Driving"
    SET $CommuteTime = @prev
    ACTION gettext Text=`Commute: {$CommuteTime} driving to {$FirstTitle} at {$FirstLocation}.`
    SET $CommuteSummary = @prev
  ELSE
    ACTION gettext Text="Commute: No location set for your first event."
    SET $CommuteSummary = @prev
  ENDIF

ELSE
  ACTION gettext Text="Calendar: No events scheduled for today."
  SET $CalendarSummary = @prev
  ACTION gettext Text=""
  SET $EventsList = @prev
  ACTION gettext Text="Commute: No events to commute to."
  SET $CommuteSummary = @prev
ENDIF

# Step 5: Build comprehensive briefing
ACTION gettext Text=`Good morning! Here is your briefing for {$DateStr}.\n\n{$WeatherSummary}\n\n{$CalendarSummary}{$EventsList}\n\n{$CommuteSummary}\n\nHave a great day!`
SET $Briefing = @prev

# Step 6: Speak the briefing aloud
ACTION speaktext WFText=$Briefing

# Step 7: Also show the briefing as text
ACTION showresult Text=$Briefing
ENDSHORTCUT
