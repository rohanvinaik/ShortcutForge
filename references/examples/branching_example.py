"""
Branching Example: API call with error handling and conditional logic
=====================================================================
Demonstrates: URL with interpolation, HTTP POST, conditionals (if/else),
variable wiring, multiple dictionary accesses, and notifications.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from shortcuts_compiler import (
    Shortcut, actions, ref_extension_input, ref_variable, wrap_token_string
)

s = Shortcut("Smart Logger", input_types=["string"])

# Step 1: Build URL using the shortcut's input (passed via share sheet)
s.add(actions.make("comment", WFCommentActionText="Build API URL with the input key"))
url = s.add(actions.make("url",
    WFURLActionURL=wrap_token_string("https://api.example.com/log?key=", ref_extension_input())))

# Step 2: POST to the API
response = s.add(actions.make("downloadurl", WFHTTPMethod="POST"))

# Step 3: Parse the response
data = s.add(actions.make("detect.dictionary"))
s.add(actions.make("setvariable", WFVariableName="ResponseData", WFInput=data))

# Step 4: Extract status
response_var = s.add(actions.make("getvariable", WFVariable=ref_variable("ResponseData")))
status = s.add(actions.make("getvalueforkey",
    WFDictionaryKey="status", WFInput=response_var))

# Step 5: Branch on whether status exists
with s.if_else_block(status, condition="has_any_value") as otherwise:
    # If branch: status exists — extract the message and notify success
    msg_var = s.add(actions.make("getvariable", WFVariable=ref_variable("ResponseData")))
    message = s.add(actions.make("getvalueforkey",
        WFDictionaryKey="message", WFInput=msg_var))
    s.add(actions.make("notification",
        WFNotificationActionTitle="Log Success",
        WFNotificationActionBody="Data logged successfully"))

    otherwise()

    # Else branch: no status — something went wrong
    s.add(actions.make("notification",
        WFNotificationActionTitle="Log Failed",
        WFNotificationActionBody="No status in response"))

# Step 6: Always show a completion notification
s.add(actions.make("notification",
    WFNotificationActionTitle="Done",
    WFNotificationActionBody="Smart Logger finished"))

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
filepath = s.save(os.path.join(output_dir, "smart_logger.shortcut"))
print(f"Saved {filepath} (unsigned)")
print(f"Actions: {len(s.actions)}")
