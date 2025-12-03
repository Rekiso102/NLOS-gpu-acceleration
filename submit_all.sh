#!/bin/bash
# Submit SLURM jobs for all 10 input files
# Each job will process one file (FILE_ID from 1 to 10)

echo "Submitting SLURM jobs for all input files..."

for id in {1..10}; do
    echo "Submitting job for FILE_ID=$id"
    sbatch --export=FILE_ID=$id main_focusing_accelerated.slurm
done

echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "View output files: ls -lh nlos_reconstruction_*.output"
