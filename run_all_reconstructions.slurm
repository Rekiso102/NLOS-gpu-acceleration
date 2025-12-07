#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=nlos_all_reconstructions.output
#SBATCH --error=nlos_all_reconstructions.error

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/12.5.0
module load conda/miniforge
bootstrap-conda

# Remove existing environment and create new one with correct Python version
echo "Removing existing nlos_env environment if it exists..."
conda env remove -n nlos_env -y 2>/dev/null || true

echo "Creating new nlos_env environment with Python 3.11..."
conda create -n nlos_env python=3.11 -y
conda activate nlos_env

echo "Installing core packages..."
conda install numpy scipy matplotlib -y

echo "Installing additional packages..."
conda install -c conda-forge mat73 cupy -y

echo "Environment setup complete. Running both scripts for all IDs..."

# Loop through all 10 scene IDs
for id in {1..10}; do
    echo ""
    echo "========== SCENE ID $id =========="

    echo "Running original version..."
    python main_focusing.py $id

    echo "Running GPU accelerated version..."
    python main_focusing_accelerated.py $id

    echo "Scene $id complete."
done

echo "========== ALL RECONSTRUCTIONS COMPLETE =========="