#!/bin/bash
#SBATCH --job-name=serial_fractal_program   # Job name
#SBATCH --output=serial_job_output.txt      # Output file for stdout
#SBATCH --error=serial_job_error.txt        # Error file for stderr
#SBATCH -N 1                                # Number of CPU cores per task
#SBATCH -p RM                               # Partition (queue) to submit to
#SBATCH -t 00:30:00                         # Max runtime (hh:mm:ss)

# Load any necessary modules (e.g., for software packages)
module load nvhpc/22.9
module load cuda/11.7.1
# module load opencv/4.2.0
# module unload opencv/4.2.0 // might need to do this. idk?
# module load openmpi/4.0.5-nvhpc22.9

# compile codes
nvcc -o Fractals.o -c Fractals.cu 
nvc++ -o frac_serial.o -lcudart -cuda -c frac_serial.cpp

# link codes together
nvc++ -cuda frac_serial.o Fractals.o -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lcudart -o main

# run executable
time $(BIN_DIR)/frac_serial


