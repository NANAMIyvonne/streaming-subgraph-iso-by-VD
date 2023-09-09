# Variables
MKLROOT=/data/fangyuec/intel/oneapi/mkl/2023.1.0
FAISS_INCLUDE=-I/data/fangyuec/faiss/faiss -I/data/fangyuec/faiss/build/
FAISS_LIB=-I$(HOME)/local/include -L$(HOME)/local/lib -lfaiss -L/data/fangyuec/cuda-12.1/lib64
LIB_MKL=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
CUDA_INCLUDE=
COMPILER=g++
FLAGS=-pthread -fopenmp -L/home/fangyuec/local/lib -lblas

# Final lib paths
LIB=$(FAISS_INCLUDE) $(CUDA_INCLUDE) $(LIB_MKL) $(FAISS_LIB)

# .cpp source files and their object files
CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp,%.o,$(CPP_FILES))

# Targets
all: main

main: $(OBJ_FILES)
	$(COMPILER) $(FLAGS) -o $@ $^ $(LIB)

%.o: %.cpp
	$(COMPILER) $(FLAGS) -c -o $@ $< $(LIB)

clean:
	rm -f *.o main
