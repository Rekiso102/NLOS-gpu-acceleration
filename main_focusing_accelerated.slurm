#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=nlos_reconstruction.output
#SBATCH --error=nlos_reconstruction.error

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/12.5.0
module load conda/miniforge
bootstrap-conda

# Create or activate conda environment
if conda env list | grep -q "nlos_env"; then
    echo "Activating existing nlos_env environment..."
    conda activate nlos_env
else
    echo "Creating new nlos_env environment..."
    conda create -n nlos_env python=3.11 -y
    conda activate nlos_env
    echo "Installing core packages..."
    conda install numpy scipy matplotlib -y
    echo "Installing additional packages..."
    conda install -c conda-forge mat73 cupy -y
fi

echo "Environment setup complete. Running Python script..."

python main_focusing_accelerated.py