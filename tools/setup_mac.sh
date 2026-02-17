#!/bin/bash
# Apple Shortcuts Compiler — One-Time macOS Setup
# Run this once to configure your Mac for seamless shortcut generation.

set -e
echo "=== Apple Shortcuts Compiler Setup ==="
echo ""

# 1. Check macOS version
sw_vers_major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$sw_vers_major" -lt 12 ]; then
  echo "ERROR: macOS 12 (Monterey) or later required for shortcuts CLI."
  echo "Your version: $(sw_vers -productVersion)"
  exit 1
fi
echo "✓ macOS $(sw_vers -productVersion) detected"

# 2. Check shortcuts CLI
if ! command -v shortcuts &> /dev/null; then
  echo "ERROR: 'shortcuts' CLI not found. It should come with macOS 12+."
  echo "Try: xcode-select --install"
  exit 1
fi
echo "✓ shortcuts CLI available"

# 3. Test signing capability
TMPFILE=$(mktemp /tmp/test_shortcut.XXXXXX.shortcut)
python3 -c "
import plistlib
data = {
    'WFWorkflowActions': [],
    'WFWorkflowClientRelease': '4.0',
    'WFWorkflowClientVersion': '2302.0.4',
    'WFWorkflowIcon': {'WFWorkflowIconGlyphNumber': 59653, 'WFWorkflowIconStartColor': 4274264319},
    'WFWorkflowImportQuestions': [],
    'WFWorkflowInputContentItemClasses': [],
    'WFWorkflowMinimumClientVersion': 900,
    'WFWorkflowMinimumClientVersionString': '900',
    'WFWorkflowTypes': ['NCWidget', 'WatchKit'],
}
with open('$TMPFILE', 'wb') as f:
    plistlib.dump(data, f)
" 2>/dev/null

SIGNED_TMP="${TMPFILE%.shortcut}_signed.shortcut"
if shortcuts sign -i "$TMPFILE" -o "$SIGNED_TMP" -m anyone 2>/dev/null; then
  echo "✓ Shortcut signing works"
  rm -f "$TMPFILE" "$SIGNED_TMP"
else
  echo "WARNING: Signing test failed. You may need to:"
  echo "  1. Open Shortcuts app at least once"
  echo "  2. Sign into iCloud"
  echo "  3. Enable Private Sharing in Shortcuts settings"
  rm -f "$TMPFILE" "$SIGNED_TMP"
fi

# 4. Create convenience directories
SIGN_INBOX="$HOME/Desktop/sign_inbox"
SIGN_OUTBOX="$HOME/Desktop/sign_outbox"
mkdir -p "$SIGN_INBOX" "$SIGN_OUTBOX"
echo "✓ Created signing folders:"
echo "  Drop unsigned → $SIGN_INBOX"
echo "  Get signed   ← $SIGN_OUTBOX"

# 5. Check for fswatch (optional, for auto-signing)
if command -v fswatch &> /dev/null; then
  echo "✓ fswatch available (auto-signing supported)"
elif command -v brew &> /dev/null; then
  echo "  fswatch not found. Install for auto-signing: brew install fswatch"
else
  echo "  fswatch not found. Install Homebrew first: https://brew.sh"
fi

# 6. Create quick-sign alias
SHELL_RC="$HOME/.zshrc"
if [ -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ]; then
  SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "sign-shortcut" "$SHELL_RC" 2>/dev/null; then
  echo "" >> "$SHELL_RC"
  echo '# Apple Shortcuts quick-sign alias' >> "$SHELL_RC"
  echo 'sign-shortcut() { shortcuts sign -i "$1" -o "${1%.shortcut}_signed.shortcut" -m "${2:-anyone}"; }' >> "$SHELL_RC"
  echo "✓ Added 'sign-shortcut' alias to $SHELL_RC"
  echo "  Usage: sign-shortcut myfile.shortcut [anyone|people-who-know-me]"
else
  echo "✓ sign-shortcut alias already exists"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick reference:"
echo "  sign-shortcut my.shortcut          → Sign for anyone"
echo "  sign-shortcut my.shortcut people-who-know-me  → Sign for contacts only"
echo "  Drop files in ~/Desktop/sign_inbox → Use auto_sign.sh for batch"
echo ""
echo "Prerequisites on each device:"
echo "  macOS: Shortcuts → Settings → Private Sharing → ON"
echo "  iOS:   Settings → Shortcuts → Private Sharing → ON"
