# Load any necessary modules (e.g., for software packages)
module load nvhpc/22.9
module load cuda/11.7.1
# module load opencv/4.2.0 # ruins linking process
# module unload opencv/4.2.0 # might need to do this. idk?
module load openmpi/4.0.5-nvhpc22.9 # downgraded by opencv/4.2.0
