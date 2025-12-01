#!/bin/bash

echo "================================================"
echo "Diabetes Risk Assessment Tool - Quick Start"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 detected: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 detected"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "================================================"
    echo "🚀 Launching Diabetes Risk Assessment Tool..."
    echo "================================================"
    echo ""
    echo "The app will open automatically in your browser at:"
    echo "👉 http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Launch Streamlit
    streamlit run diabetes_risk_tool.py
else
    echo ""
    echo "❌ Failed to install dependencies."
    echo "Please check your internet connection and try again."
    exit 1
fi
