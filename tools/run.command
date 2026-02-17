#!/bin/bash
# ShortcutForge — Double-click to start the web UI
# Opens http://localhost:8000 in your browser

cd "$(dirname "$0")"

echo ""
echo "  ┌─────────────────────────────┐"
echo "  │      ShortcutForge          │"
echo "  │  Natural Language → Shortcut │"
echo "  └─────────────────────────────┘"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "  ⚠️  ANTHROPIC_API_KEY not set."
    echo ""
    echo "  Set it with:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
    echo "  Or add to ~/.zshrc / ~/.bashrc for persistence."
    echo ""
    read -p "  Enter your API key (or press Enter to skip): " key
    if [ -n "$key" ]; then
        export ANTHROPIC_API_KEY="$key"
        echo "  ✓ Key set for this session."
    else
        echo "  Continuing without API key — DSL compilation will work, but LLM generation won't."
    fi
    echo ""
fi

# Install dependencies quietly if missing
echo "  Checking dependencies..."
pip3 install -q lark anthropic fastapi uvicorn 2>/dev/null
echo "  ✓ Dependencies ready."
echo ""

# Open browser after a short delay
(sleep 2 && open http://localhost:8000) &

# Start the server
echo "  Starting server at http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo ""
python3 scripts/server.py
