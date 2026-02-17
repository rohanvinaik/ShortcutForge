SHORTCUT "Photo Metadata Report"
# Media Metadata Pipeline
# Select photos, extract EXIF data, build a summary report, share or save

# Step 1: Prompt user to select photos
ACTION selectphoto WFSelectMultiplePhotos=true
SET $Photos = @prev

IF $Photos has_any_value
  # Step 2: Count the selected photos
  ACTION count WFInput=$Photos
  SET $PhotoCount = @prev

  # Step 3: Initialize the report header
  ACTION date WFDateActionMode="Current Date"
  SET $Today = @prev
  ACTION format.date WFDateFormatStyle="Medium" WFInput=$Today
  SET $DateStr = @prev

  ACTION gettext Text=`Photo Metadata Report - {$DateStr}\n===================================\nPhotos analyzed: {$PhotoCount}\n`
  SET $Report = @prev

  # Step 4: Loop through each photo and extract metadata
  SET $Index = 0
  FOREACH $Photos
    # Increment index
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$Index
    SET $Index = @prev

    # Get image width
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Width"
    SET $Width = @prev

    # Get image height
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Height"
    SET $Height = @prev

    # Get date taken
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Date Taken"
    SET $DateTaken = @prev

    # Get camera make
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Camera Make"
    SET $CameraMake = @prev

    # Get camera model
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Camera Model"
    SET $CameraModel = @prev

    # Get orientation
    ACTION properties.images WFInput=@item WFContentItemPropertyName="Orientation"
    SET $Orientation = @prev

    # Build entry for this photo
    ACTION gettext Text=`\nPhoto {$Index}:\n  Dimensions: {$Width} x {$Height}\n  Date Taken: {$DateTaken}\n  Camera: {$CameraMake} {$CameraModel}\n  Orientation: {$Orientation}\n---`
    SET $Entry = @prev

    # Append entry to the report
    ACTION gettext Text=`{$Report}{$Entry}`
    SET $Report = @prev
  ENDFOREACH

  # Step 5: Add summary footer
  ACTION gettext Text=`{$Report}\n\n===================================\nEnd of Report - {$PhotoCount} photos analyzed`
  SET $Report = @prev

  # Step 6: Offer to share or save
  MENU "What would you like to do with the report?"
  CASE "View Report"
    ACTION showresult Text=$Report
  CASE "Share Report"
    ACTION share WFInput=$Report
  CASE "Save to Note"
    ACTION appendnote WFInput=$Report WFNote="Photo Reports"
    ACTION showresult Text="Report saved to Photo Reports note."
  ENDMENU

ELSE
  ACTION alert WFAlertActionMessage="No photos were selected. Please try again and select at least one photo." WFAlertActionTitle="No Photos"
ENDIF
ENDSHORTCUT
