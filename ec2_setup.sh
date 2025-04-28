#!/bin/bash
# EC2 G5 Instance Setup Script for Bug Detection System

# Update package information
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    xvfb \
    x11-utils \
    mesa-utils \
    libgl1-mesa-glx

# Set up a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install CUDA support for PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Configure X virtual framebuffer for screenshot capabilities
echo "Setting up Xvfb for headless operation..."
Xvfb :1 -screen 0 1280x1024x24 &
export DISPLAY=:1

echo "Setup complete!"
