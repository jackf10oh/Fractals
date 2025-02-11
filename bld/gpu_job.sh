#!/bin/bash
#SBATCH --job-name=gpu_fractal_job            # Job name
#SBATCH --output=gpu_job_output.txt           # Output file for stdout
#SBATCH --error=gpu_job_error.txt             # Error file for stderr
#SBATCH -N 2                                  # Number of CPU cores per task
#SBATCH -p GPU-small                          # Partition (queue) to submit to
#SBATCH -gres=gpu:v100-16:8                   # general resource = gpu : gpu_type: gpus_per_core
#SBATCH -t 00:05:00                           # Max runtime (hh:mm:ss)

# Load any necessary modules (e.g., for software packages)
module load nvhpc/22.9
module load cuda/11.7.1
# module load opencv/4.2.0 # ruins linking process
# module unload opencv/4.2.0 // might need to do this. idk?

make $(BIN_DIR)/frac_cuda

# run executable
time $(BIN_DIR)/frac_cuda


