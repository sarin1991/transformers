#!/bin/bash

# Setup script for Cast Triton kernels
echo "🚀 Setting up Cast Triton kernels environment..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA not found. Triton kernels require CUDA to work properly."
    echo "   You can still install the packages, but the kernels won't work without CUDA."
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "📥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
echo "📥 Installing Triton..."
pip install triton

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import triton; print(f'Triton version: {triton.__version__}')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the Triton kernel test:"
echo "  python triton_kernels.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate" 