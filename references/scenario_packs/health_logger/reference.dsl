SHORTCUT "Log Health"
# Health Logger Companion Shortcut
# Receives a 6-character key, fetches nutrient data from the
# Health Logger worker, and logs each nutrient to Apple HealthKit.

# Step 1: Get the input key (from Shortcut Input or ask user)
IF @input has_any_value
  SET $Key = @input
ELSE
  ACTION ask Text="Enter your 6-character health log key:"
  SET $Key = @prev
ENDIF

# Step 2: Build the fetch URL
ACTION gettext Text="https://health-logger.rohan-vinaik.workers.dev/fetch?key="
SET $BaseURL = @prev
ACTION gettext Text=`{$BaseURL}{$Key}`
SET $FetchURL = @prev

# Step 3: Fetch the nutrient data from the server
ACTION url WFURLActionURL=$FetchURL
ACTION downloadurl WFHTTPMethod="GET"
SET $Response = @prev

# Step 4: Check if we got valid data
IF $Response has_any_value
  # Parse the response as a dictionary
  ACTION detect.dictionary
  SET $NutrientDict = @prev

  # Step 5: Initialize a counter for logged nutrients
  SET $LogCount = 0

  # Step 6: Log Caffeine if present
  ACTION getvalueforkey WFDictionaryKey="caf" WFInput=$NutrientDict
  SET $CafValue = @prev
  IF $CafValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Caffeine" WFQuantitySampleQuantity=$CafValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 7: Log Vitamin C if present
  ACTION getvalueforkey WFDictionaryKey="vc" WFInput=$NutrientDict
  SET $VCValue = @prev
  IF $VCValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Vitamin C" WFQuantitySampleQuantity=$VCValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 8: Log Vitamin D if present
  ACTION getvalueforkey WFDictionaryKey="vd" WFInput=$NutrientDict
  SET $VDValue = @prev
  IF $VDValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Vitamin D" WFQuantitySampleQuantity=$VDValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 9: Log Folate if present
  ACTION getvalueforkey WFDictionaryKey="fol" WFInput=$NutrientDict
  SET $FolValue = @prev
  IF $FolValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Folate, DFE" WFQuantitySampleQuantity=$FolValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 10: Log Vitamin B12 if present
  ACTION getvalueforkey WFDictionaryKey="b12" WFInput=$NutrientDict
  SET $B12Value = @prev
  IF $B12Value has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Vitamin B12" WFQuantitySampleQuantity=$B12Value
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 11: Log Vitamin B6 if present
  ACTION getvalueforkey WFDictionaryKey="b6" WFInput=$NutrientDict
  SET $B6Value = @prev
  IF $B6Value has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Vitamin B6" WFQuantitySampleQuantity=$B6Value
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 12: Log Iron if present
  ACTION getvalueforkey WFDictionaryKey="fe" WFInput=$NutrientDict
  SET $FeValue = @prev
  IF $FeValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Iron" WFQuantitySampleQuantity=$FeValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 13: Log Zinc if present
  ACTION getvalueforkey WFDictionaryKey="zn" WFInput=$NutrientDict
  SET $ZnValue = @prev
  IF $ZnValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Zinc" WFQuantitySampleQuantity=$ZnValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 14: Log Magnesium if present
  ACTION getvalueforkey WFDictionaryKey="mg" WFInput=$NutrientDict
  SET $MgValue = @prev
  IF $MgValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Magnesium" WFQuantitySampleQuantity=$MgValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 15: Log Calcium if present
  ACTION getvalueforkey WFDictionaryKey="ca" WFInput=$NutrientDict
  SET $CaValue = @prev
  IF $CaValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Calcium" WFQuantitySampleQuantity=$CaValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 16: Log Selenium if present
  ACTION getvalueforkey WFDictionaryKey="se" WFInput=$NutrientDict
  SET $SeValue = @prev
  IF $SeValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Selenium" WFQuantitySampleQuantity=$SeValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 17: Log Protein if present
  ACTION getvalueforkey WFDictionaryKey="pro" WFInput=$NutrientDict
  SET $ProValue = @prev
  IF $ProValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Protein" WFQuantitySampleQuantity=$ProValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 18: Log Calories if present
  ACTION getvalueforkey WFDictionaryKey="cal" WFInput=$NutrientDict
  SET $CalValue = @prev
  IF $CalValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Active Energy Burned" WFQuantitySampleQuantity=$CalValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 19: Log Carbohydrates if present
  ACTION getvalueforkey WFDictionaryKey="carb" WFInput=$NutrientDict
  SET $CarbValue = @prev
  IF $CarbValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Carbohydrates" WFQuantitySampleQuantity=$CarbValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 20: Log Total Fat if present
  ACTION getvalueforkey WFDictionaryKey="fat" WFInput=$NutrientDict
  SET $FatValue = @prev
  IF $FatValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Total Fat" WFQuantitySampleQuantity=$FatValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 21: Log Fiber if present
  ACTION getvalueforkey WFDictionaryKey="fib" WFInput=$NutrientDict
  SET $FibValue = @prev
  IF $FibValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Fiber" WFQuantitySampleQuantity=$FibValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 22: Log Sodium if present
  ACTION getvalueforkey WFDictionaryKey="sod" WFInput=$NutrientDict
  SET $SodValue = @prev
  IF $SodValue has_any_value
    ACTION health.quantity.log WFQuantitySampleType="Sodium" WFQuantitySampleQuantity=$SodValue
    ACTION math WFMathOperation="+" WFMathOperand=1 WFInput=$LogCount
    SET $LogCount = @prev
  ENDIF

  # Step 23: Show completion summary
  ACTION showresult Text=`Successfully logged {$LogCount} nutrients to Apple Health!`

ELSE
  # No data returned - show error
  ACTION alert WFAlertActionMessage="No nutrient data found for this key. The key may have expired (keys expire after 1 hour). Please try logging again from the Health Logger app." WFAlertActionTitle="No Data Found"
ENDIF
ENDSHORTCUT
