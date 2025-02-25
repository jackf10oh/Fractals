#!/bin/bash
#SBATCH --job-name=mpi_fractal_job            # Job name
#SBATCH --output=out/mpi_job_output.txt       # Output file for stdout
#SBATCH --error=out/mpi_job_error.txt         # Error file for stderr
#SBATCH -N 4                                  # Number of CPU cores per task
#SBATCH -p GPU-shared                         # Partition (queue) to submit to
#SBATCH --gres=gpu:v100-32:4                  # general resource = gpu : gpu_type: gpus_per_core
#SBATCH -t 00:05:00                           # Max runtime (hh:mm:ss)

# Load any necessary modules (e.g., for software packages)
module load nvhpc/22.9
module load cuda/11.7.1
module load openmpi/4.0.5-nvhpc22.9
# module load opencv/4.2.0
# module unload opencv/4.2.0 // might need to do this. idk?

# run executable
time mpirun -n 5 ${BIN_DIR}/frac_mpi
