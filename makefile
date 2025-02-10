-include $(OBJ_DIR)/*.d

# Compilers and flags
NVCC = nvcc  
NVCCFLAGS = -lcudart -cuda
NVCXX = nvc++
NVCXXFLAGS = -lcudart -cuda -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
MPICXX = mpic++
MPICXXFLAGS = -lcudart -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

# Directories 
BLD_DIR = ./bld
SRC_DIR = ./src 
OBJ_DIR = ./obj 
BIN_DIR = ./bin  

# Source Files
SRC = $(SRC_DIR)/frac_serial.cpp $(SRC_DIR)/frac_mpi.cpp $(SRC_DIR)/frac_cuda.cpp
CUDA_SRC = $(SRC_DIR)/Fractals.cu

# Object Files
OBJ = $(OBJ_DIR)/frac_serial.o $(OBJ_DIR)/frac_mpi.o $(OBJ_DIR)/frac_cuda.o $(OBJ_DIR)/fractals.o
EXEC = $(BIN_DIR)/frac_serial $(BIN_DIR)/frac_mpi $(BIN_DIR)/frac_cuda

# Slurm SBATCH job scripts 
GPU_SBATCH_SCRIPT = $(BLD_DIR)/gpi_job.sh
MPI_SBATCH_SCRIPT = $(BLD_DIR)/mpi_job.sh
SERIAL_SBATCH_SCRIPT = $(BLD_DIR)/serial_job.sh

# Create necessary directories (if they do not exist)
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile .cpp file with NVCXX for the CUDA version
$(OBJ_DIR)/frac_cuda.o: $(SRC_DIR)/frac_cuda.cpp
	$(NVCXX) -o $@ $(NVCXXFLAGS) -I$(SRC_DIR) -MMD -MP -c $(SRC_DIR)/$<

# Compile .cpp file with MPICXX for the MPI version
$(OBJ_DIR)/frac_mpi.o: $(SRC_DIR)/frac_mpi.cpp
	$(MPICXX) -o $@ $(MPICXXFLAGS) -c -I$(SRC_DIR) -MMD -MP $(SRC_DIR)/$<

# Compile .cpp file with NVCXX for the serial version
$(OBJ_DIR)/frac_serial.o: $(SRC_DIR)/frac_serial.cpp
	$(NVCXX) -o $@ $(NVCXXFLAGS) -c -I$(SRC_DIR) -MMD -MP $(SRC_DIR)/$<

# Compile Fractals.cu to fractals.o using NVCC
$(OBJ_DIR)/fractals.o: $(SRC_DIR)/Fractals.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -I$(SRC_DIR) -MMD -MP -c $(SRC_DIR)/$<


# Link object files into executables
$(BIN_DIR)/frac_serial: $(OBJ_DIR)/frac_serial.o $(OBJ_DIR)/fractals.o
	$(NVCXX) -o $@ $(NVCXXFLAGS) $^ 

$(BIN_DIR)/frac_mpi: $(OBJ_DIR)/frac_mpi.o $(OBJ_DIR)/fractals.o
	$(MPICXX) -o $@ $(MPICXXFLAGS) $^ 

$(BIN_DIR)/frac_cuda: $(OBJ_DIR)/frac_cuda.o $(OBJ_DIR)/fractals.o
	$(NVCXX) -o $@ $(NVCXXFLAGS) $^ 

# Target to load necessary modules
setup_modules:
	bash ./bld/setup_modules.sh

# Default target (build everything)
all: $(EXEC) setup_modules

submit_gpu_job:
	@echo "Submitting a gpu job using sbatch..."
	$(MAKE) $(BIN_DIR)/frac_cuda
	$(shell sbatch $(GPU_SBATCH_SCRIPT))

submit_mpi_job:
	@echo "Submitting an mpi job using sbatch..."
	$(MAKE) $(BIN_DIR)/frac_mpi
	$(shell sbatch $(MPI_SBATCH_SCRIPT))

submit_serial_job:
	@echo "Submitting a serial job using sbatch..."
	$(MAKE) $(BIN_DIR)/frac_serial
	$(shell sbatch $(SERIAL_SBATCH_SCRIPT))

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

