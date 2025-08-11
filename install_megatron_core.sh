#!/bin/bash

# Script to install megatron-core from AMD version
# This script converts the Docker RUN command into a standalone shell script

set -e  # Exit on any error

echo "Starting megatron-core installation..."

# Uninstall existing megatron-core if present
echo "Uninstalling existing megatron-core..."
pip uninstall -y megatron-core || true

# Clone the AMD version of Megatron-LM
echo "Cloning Megatron-LM AMD version..."
git clone https://github.com/yushengsu-thu/Megatron-LM-amd_version.git

# Change to the cloned directory
echo "Changing to Megatron-LM-amd_version directory..."
cd Megatron-LM-amd_version

# Install the package in editable mode with verbose output
echo "Installing megatron-core in editable mode..."
pip install -vvv -e .

# Return to the original directory (equivalent to cd /workspace/ in Docker)
echo "Returning to original directory..."
cd ..

echo "megatron-core installation completed successfully!" 