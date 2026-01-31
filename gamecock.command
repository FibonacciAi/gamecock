#!/bin/bash
# GAMECOCK - SEC Filing Truth Finder
# Double-click to run on macOS

cd "$(dirname "$0")"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    echo "Install from: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "============================================"
echo "  GAMECOCK - SEC Filing Truth Finder"
echo "============================================"
echo ""

# Run the main script
python3 gamecock.py

read -p "Press Enter to exit..."
