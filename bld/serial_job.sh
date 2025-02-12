#!/bin/bash
#SBATCH --job-name=serial_fractal_program   # Job name
#SBATCH --output=out/serial_job_output.txt      # Output file for stdout
#SBATCH --error=out/serial_job_error.txt        # Error file for stderr
#SBATCH -N 1                                # Number of CPU cores per task
#SBATCH -p RM                               # Partition (queue) to submit to
#SBATCH -t 01:15:00                         # Max runtime (hh:mm:ss)

# Load any necessary modules (e.g., for software packages)
module load nvhpc/22.9
module load cuda/11.7.1
# module load opencv/4.2.0 # ruins linking process
# module unload opencv/4.2.0 # might need to do this. idk?

# run executable
time ${BIN_DIR}/frac_serial


