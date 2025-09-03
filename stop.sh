#!/bin/bash
# Smart Second Brain - System Stop Script
# Simple wrapper for the Python stop script

echo "🛑 Smart Second Brain - System Stop"
echo "==================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not available"
    exit 1
fi

# Check if psutil is available (required for stop script)
/opt/miniconda3/bin/python -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: psutil package is not available"
    echo "   Install with: pip install psutil"
    exit 1
fi

echo "✅ Environment check passed"
echo "🛑 Stopping system components..."
echo ""

# Run the Python stop script
/opt/miniconda3/bin/python stop_system.py

# Exit with the same code as the Python script
exit $?
