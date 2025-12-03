#!/bin/bash
# One-time setup script to create the conda environment
# Run this once before submitting jobs with submit_all.sh

echo "Setting up nlos_env environment..."

module load nvidia/cuda/12.5.0
module load conda/miniforge
bootstrap-conda

# Check if environment already exists
if conda env list | grep -q "^nlos_env "; then
    echo "Environment nlos_env already exists!"
    echo "To recreate, first run: conda env remove -n nlos_env -y"
    exit 0
fi

echo "Creating new nlos_env environment with Python 3.11..."
conda create -n nlos_env python=3.11 -y
conda activate nlos_env

echo "Installing core packages..."
conda install numpy scipy matplotlib -y

echo "Installing additional packages..."
conda install -c conda-forge mat73 cupy -y

echo ""
echo "✓ Environment setup complete!"
echo "You can now run: ./submit_all.sh"
