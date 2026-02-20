"""
Linear Example: Fetch weather data and show it
================================================
Demonstrates: URL, HTTP GET, JSON parsing, dictionary access, show result.
No branching, no variables — just a straight pipeline.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
)
from shortcuts_compiler import Shortcut, actions, ref_variable

s = Shortcut("Weather Check")

# Step 1: Comment
s.add(actions.make("comment", WFCommentActionText="Fetch weather data from API"))

# Step 2: Build the API URL
url = s.add(
    actions.make(
        "url", WFURLActionURL="https://api.weather.example.com/current?city=Seattle"
    )
)

# Step 3: Download the JSON (GET is default)
response = s.add(actions.make("downloadurl"))

# Step 4: Parse JSON into dictionary
data = s.add(actions.make("detect.dictionary"))

# Step 5: Store it for multi-key access
s.add(actions.make("setvariable", WFVariableName="WeatherData", WFInput=data))

# Step 6: Get the variable back, then extract temperature
weather = s.add(actions.make("getvariable", WFVariable=ref_variable("WeatherData")))
temp = s.add(
    actions.make("getvalueforkey", WFDictionaryKey="temperature", WFInput=weather)
)

# Step 7: Show the result
s.add(
    actions.make(
        "showresult", Text=temp.in_string("Current temperature: \ufffc\u00b0F")
    )
)

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
filepath = s.save(os.path.join(output_dir, "weather_check.shortcut"))
print(f"Saved {filepath} (unsigned)")
print(f"Actions: {len(s.actions)}")
