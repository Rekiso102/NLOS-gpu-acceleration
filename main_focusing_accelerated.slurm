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
module load python/3.11.6

if [ ! -d "nlos_env" ]; then
    python3 -m venv nlos_env
    source nlos_env/bin/activate
    pip install --upgrade pip
    pip install numpy scipy matplotlib mat73
    pip install cupy-cuda12x
else
    source nlos_env/bin/activate
fi

python3 main_focusing_accelerated.py