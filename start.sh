#!/bin/bash
# Smart Second Brain - System Startup Script
# Simple wrapper for the Python startup script

echo "🧠 Smart Second Brain - System Startup"
echo "======================================"

# Check if we're in the right directory
if [ ! -d "api" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: This script must be run from the project root directory"
    echo "   Expected: api/ and frontend/ folders"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not available"
    exit 1
fi

# Check if virtual environment is activated (optional)
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your conda environment: conda activate base"
fi

echo "✅ Environment check passed"
echo "🚀 Starting system components..."
echo ""

# Run the Python startup script
/opt/miniconda3/bin/python start_system.py

# Exit with the same code as the Python script
exit $?
