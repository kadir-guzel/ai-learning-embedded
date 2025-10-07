#!/bin/bash

# AI Learning Workspace Setup Script
# ==================================

set -e  # Exit on any error

echo "🚀 Setting up AI Learning Workspace"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv ai_env
    source ai_env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional packages
echo "📦 Installing additional packages..."
python3 -m pip install -r requirements.txt

# Test the installation
echo "🧪 Testing installation..."
python3 test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run examples: python3 01-fastai-basics/simple_example.py"
echo "2. Compare frameworks: python3 02-pytorch-equivalent/pytorch_classifier.py"
echo "3. Explore advanced topics in other directories"
echo ""
echo "Happy learning! 🤖"