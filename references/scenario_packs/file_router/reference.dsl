SHORTCUT "File Router"
# File Router
# Accept a file, detect its type/extension, and route it
# to the appropriate handler: images get resized, PDFs get
# previewed, text files get shared, others get Quick Looked.

# Step 1: Get the input file
IF @input has_any_value
  SET $InputFile = @input
ELSE
  ACTION file.select
  SET $InputFile = @prev
ENDIF

# Step 2: Get the file extension
ACTION properties.files WFContentItemPropertyName="File Extension" WFInput=$InputFile
SET $Extension = @prev

# Step 3: Get the file name for display
ACTION properties.files WFContentItemPropertyName="Name" WFInput=$InputFile
SET $FileName = @prev

# Step 4: Get file size for display
ACTION properties.files WFContentItemPropertyName="File Size" WFInput=$InputFile
SET $FileSize = @prev

# Step 5: Route based on file extension using MENU
MENU `Route: {$FileName} ({$Extension})`
  CASE "Image"
    # Step 5a: Handle image files - resize
    ACTION image.resize WFImageResizeWidth=1024 WFImage=$InputFile
    SET $Resized = @prev

    # Save resized image
    ACTION setclipboard WFInput=$Resized
    ACTION showresult Text=`Resized image {$FileName} to 1024px width. Copied to clipboard.`

  CASE "PDF"
    # Step 5b: Handle PDF files - preview with Quick Look
    ACTION previewdocument WFInput=$InputFile
    ACTION showresult Text=`Previewed PDF: {$FileName} ({$FileSize})`

  CASE "Text"
    # Step 5c: Handle text files - get and display content
    ACTION gettext Text=`{$InputFile}`
    SET $TextContent = @prev

    ACTION count WFCountType="Characters" WFInput=$TextContent
    SET $CharCount = @prev

    ACTION showresult Text=`Text file: {$FileName}\n{$CharCount} characters\n\n{$TextContent}`

  CASE "Other"
    # Step 5d: Handle other file types - open in appropriate app
    ACTION openin WFInput=$InputFile WFOpenInAskWhenRun=true
    ACTION showresult Text=`Opened {$FileName} ({$Extension}) in selected app.`
ENDMENU
ENDSHORTCUT
