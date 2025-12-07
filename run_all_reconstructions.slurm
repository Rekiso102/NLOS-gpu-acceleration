#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=02:00:00
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

echo "Environment setup complete."

# Create results summary file
echo "Scene_ID,Original_Time_Seconds,Accelerated_Time_Seconds,Speedup" > speedup_results.csv

echo "========== RUNNING RECONSTRUCTION FOR ALL 10 SCENE IDs =========="

# Loop through all 10 scene IDs
for id in {1..10}; do
    echo ""
    echo "==================== SCENE ID: $id ===================="

    echo "Running original version for scene $id..."
    original_output=$(python main_focusing.py $id 2>&1)
    original_time=$(echo "$original_output" | grep "Original Reconstruction Time:" | sed 's/.*: \([0-9.]*\) seconds.*/\1/')

    echo "Running GPU accelerated version for scene $id..."
    accelerated_output=$(python main_focusing_accelerated.py $id 2>&1)
    accelerated_time=$(echo "$accelerated_output" | grep "GPU Accelerated Reconstruction Time:" | sed 's/.*: \([0-9.]*\) seconds.*/\1/')

    # Calculate speedup
    if [[ -n "$original_time" && -n "$accelerated_time" && "$accelerated_time" != "0" ]]; then
        speedup=$(echo "scale=2; $original_time / $accelerated_time" | bc -l)
        echo "Scene $id: Original=${original_time}s, Accelerated=${accelerated_time}s, Speedup=${speedup}x"
        echo "$id,$original_time,$accelerated_time,$speedup" >> speedup_results.csv
    else
        echo "Scene $id: Error extracting timing data"
        echo "$id,ERROR,ERROR,ERROR" >> speedup_results.csv
    fi

    # Print the actual output for verification
    echo "Original output:"
    echo "$original_output"
    echo "Accelerated output:"
    echo "$accelerated_output"
    echo ""
done

echo "========== SUMMARY =========="
echo "Results saved to speedup_results.csv"
cat speedup_results.csv

# Calculate average speedup
echo ""
echo "Calculating average speedup..."
average_speedup=$(awk -F',' 'NR>1 && $4!="ERROR" {sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}' speedup_results.csv)
echo "Average speedup across all scenes: ${average_speedup}x"

echo "========== ALL RECONSTRUCTIONS COMPLETE =========="