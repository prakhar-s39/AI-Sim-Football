#!/bin/bash
# Installation script for AI Sim Football MAPPO dependencies

echo "Installing AI Sim Football MAPPO dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Install PyTorch with CUDA support (if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy pygame matplotlib tqdm gym stable-baselines3

echo "Installation complete!"
echo ""
echo "To test the installation, run:"
echo "python test_mappo.py"
echo ""
echo "To start training, run:"
echo "python main.py --mode train --episodes 100"
