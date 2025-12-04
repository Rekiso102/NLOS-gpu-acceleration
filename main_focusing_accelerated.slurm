#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=nlos_reconstruction_%j_id%a.output
#SBATCH --error=nlos_reconstruction_%j_id%a.error

cd $SLURM_SUBMIT_DIR

# Save FILE_ID before conda activation (conda can clear environment variables)
SAVED_FILE_ID=${FILE_ID}

module load nvidia/cuda/12.5.0
module load conda/miniforge
bootstrap-conda

# Check if environment exists, create only if needed
if conda env list | grep -q "^nlos_env2 "; then
    echo "Environment nlos_env2 already exists, activating..."
    conda activate nlos_env2
else
    echo "Creating new nlos_env2 environment with Python 3.11..."
    conda create -n nlos_env2 python=3.11 -y
    conda activate nlos_env2

    echo "Installing core packages..."
    conda install numpy scipy matplotlib -y

    echo "Installing additional packages..."
    conda install -c conda-forge mat73 cupy -y
fi

# Restore FILE_ID after conda activation
export FILE_ID=${SAVED_FILE_ID}

echo "Environment setup complete. Processing FILE_ID=${FILE_ID}..."

echo "========== GPU ACCELERATED VERSION (FILE_ID=${FILE_ID}) =========="
python main_focusing_accelerated.py

echo "========== PROCESSING COMPLETE =========="