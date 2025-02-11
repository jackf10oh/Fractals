###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=opt/packages/cuda/v11.7.1

##########################################################

## NVCC COMPILER OPTIONS ##

NVCC=nvcc
NVCC_FLAGS= 
NVCC_LIBS= -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lcudart 
# -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## NVCXX COMPILER OPTIONS:
NVCXX=nvc++
NVCXX_FLAGS =
NVCXX_LIBS = $(CUDA_LIB_DIR) -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio $(CUDA_LINK_LIBS)

##########################################################

## NVCXX COMPILER OPTIONS:
MPICXX=mpic++
MPICXX_FLAGS =
MPICXX_LIBS = $(CUDA_LIB_DIR) -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio $(CUDA_LINK_LIBS)

##########################################################

## Project file structure ##

# Include header file diretory:
INC_DIR := incl
# Source file directory:
SRC_DIR := src
# Build files directory // slurm batch files
BLD_DIR := bld
# Object file directory:
OBJ_DIR := obj
# Object file directory:
BIN_DIR := bin

# make directories if they don't exist already
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

##########################################################

## Make variables ##

# Target executable name:
EXE = $(BIN_DIR)/frac_cuda $(BIN_DIR)/frac_serial $(BIN_DIR)/frac_mpi

# Object files:
OBJS = $(OBJ_DIR)/Fractals.o $(OBJ_DIR)/frac_cuda.o $(OBJ_DIR)/frac_serial.o # $(OBJ_DIR)/frac_mpi.o

# Slurm SBATCH job scripts 
GPU_SBATCH_SCRIPT = $(BLD_DIR)/gpi_job.sh
MPI_SBATCH_SCRIPT = $(BLD_DIR)/mpi_job.sh
SERIAL_SBATCH_SCRIPT = $(BLD_DIR)/serial_job.sh
 
##########################################################

## Compile ##

# Main target: builds all
all: $(OBJ_DIR) $(BIN_DIR) $(OBJS) $(EXE)

# MPI Link .o files to executable
$(BIN_DIR)/frac_mpi : $(OBJ_DIR)/frac_mpi.o $(OBJ_DIR)/Fractals.o
	$(MPICXX) -o $@ -I$(INC_DIR) -I$(SRC_DIR) $(CUDA_INC_DIR) $(MPICXX_LIBS) -cuda $< 

# Link .o files to executable
$(BIN_DIR)/% : $(OBJ_DIR)/%.o $(OBJ_DIR)/Fractals.o
	$(NVCXX) -o $@ -I$(INC_DIR) -I$(SRC_DIR) $(CUDA_INC_DIR) $(NVCXX_LIBS) -cuda $< 

# MPI Compile C++ source files to object files:
$(OBJ_DIR)/frac_mpi.o : $(SRC_DIR)/frac_mpi.cpp $(INC_DIR)/Fractals.hpp
	$(MPICXX) -o $@ -I$(INC_DIR) -I$(SRC_DIR) $(CUDA_INC_DIR) $(MPICXX_LIBS) -cuda -c $< 

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/Fractals.hpp
	$(NVCXX) -o $@ -I$(INC_DIR) -I$(SRC_DIR) $(CUDA_INC_DIR) $(NVCXX_LIBS) -cuda -c $< 

# Compile .cu to .o using NVCC
$(OBJ_DIR)/Fractals.o: $(SRC_DIR)/Fractals.cu $(INC_DIR)/Fractals.hpp
	$(NVCC) -o $@ -I$(INC_DIR) -I$(SRC_DIR) $(NVCC_LIBS) -c $<

submit_gpu_job:
	@echo "Submitting a gpu job using sbatch..."
	make $(BIN_DIR)/frac_cuda
	@sbatch --export=BIN_DIR=$(BIN_DIR) $(GPU_SBATCH_SCRIPT)

submit_mpi_job:
	@echo "Submitting an mpi job using sbatch..."
	make $(BIN_DIR)/frac_mpi
	@sbatch --export=BIN_DIR=$(BIN_DIR) $(MPI_SBATCH_SCRIPT)

submit_serial_job:
	@echo "Submitting a serial job using sbatch..."
	make $(BIN_DIR)/frac_serial
	@sbatch --export=BIN_DIR=$(BIN_DIR) $(SERIAL_SBATCH_SCRIPT)


# Clean objects in object directory.
clean:
	$(RM) bin/* *.o obj/* *.o $(EXE)