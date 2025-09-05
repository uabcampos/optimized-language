#!/bin/bash
# Startup script for the Smart PDF Language Flagger Web Interface

echo "🚀 Starting Smart PDF Language Flagger Web Interface..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install web requirements if not already installed
echo "📦 Installing web interface dependencies..."
pip install -r requirements_web.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your OPENAI_API_KEY"
fi

# Start the advanced web interface
echo "🌐 Starting advanced web interface on http://localhost:8501"
streamlit run web_app_advanced.py --server.port 8501 --server.address localhost
