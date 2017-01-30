

# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O3

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


kmeans: calcdb.o cuda.o eqclass.o hashtree.o stats.o treeminer.o cuda_kernel.o
	$(CC) $(CC_FLAGS) calcdb.o cuda.o eqclass.o hashtree.o stats.o treeminer.o cuda_kernel.o -o gpuFTM -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

gpuFTM.o: gpuFTM.cu 
	$(NVCC) $(NVCC_FLAGS) -c gpuFTM.cu

clean:
	rm -f *.o *~ gpuFTM gpuFTM.linkinfo
