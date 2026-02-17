SHORTCUT "API Paginator"
# API Pagination Fetcher
# Fetch paginated API results, accumulate items across pages,
# format combined output, and display the result.

# Step 1: Set the base API URL
ACTION gettext Text="https://api.example.com/items"
SET $BaseURL = @prev

# Step 2: Initialize pagination variables
SET $PageNum = 1
SET $MaxPages = 10
SET $HasMore = 1

# Step 3: Initialize empty accumulator for results
ACTION gettext Text=""
SET $AllItems = @prev
SET $TotalCount = 0

# Step 4: Loop through pages using REPEAT
REPEAT 10
  # Step 4a: Build the paginated URL
  ACTION gettext Text=`{$BaseURL}?page={$PageNum}&per_page=20`
  SET $PageURL = @prev

  # Step 4b: Fetch this page
  ACTION url WFURLActionURL=$PageURL
  ACTION downloadurl WFHTTPMethod="GET"
  SET $Response = @prev

  # Step 4c: Check if response has data
  IF $Response has_any_value
    # Parse JSON response
    ACTION detect.dictionary
    SET $PageData = @prev

    # Extract items array
    ACTION getvalueforkey WFDictionaryKey="items" WFInput=$PageData
    SET $Items = @prev

    # Count items on this page
    ACTION count WFInput=$Items
    SET $PageItemCount = @prev

    # Process each item on this page
    FOREACH $Items
      # Extract item name
      ACTION getvalueforkey WFDictionaryKey="name" WFInput=@item
      SET $ItemName = @prev

      # Extract item ID
      ACTION getvalueforkey WFDictionaryKey="id" WFInput=@item
      SET $ItemID = @prev

      # Build formatted line for this item
      ACTION gettext Text=`{$ItemID}: {$ItemName}`

      # Append to accumulator
      ACTION appendvariable WFVariableName="AllItems"
    ENDFOREACH

    # Update total count
    ACTION math WFMathOperation="+" WFMathOperand=$PageItemCount WFInput=$TotalCount
    SET $TotalCount = @prev

    # Check if there are more pages
    ACTION getvalueforkey WFDictionaryKey="has_more" WFInput=$PageData
    SET $HasMore = @prev

    IF $HasMore does_not_have_any_value
      # No more pages - exit early by setting page beyond max
      SET $PageNum = 99
    ELSE
      # Increment page number
      ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$PageNum
      SET $PageNum = @prev
    ENDIF

  ELSE
    # No response - stop pagination
    SET $PageNum = 99
  ENDIF
ENDREPEAT

# Step 5: Format the combined results
ACTION text.combine WFTextSeparator="newline" WFInput=$AllItems
SET $FormattedResults = @prev

# Step 6: Build summary header
ACTION gettext Text=`=== API Results ===\nTotal items fetched: {$TotalCount}\n\n{$FormattedResults}`
SET $Output = @prev

# Step 7: Show the final output
ACTION showresult Text=`{$Output}`
ENDSHORTCUT
