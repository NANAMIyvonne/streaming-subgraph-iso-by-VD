MKLROOT=/data/fangyuec/intel/oneapi/mkl/2023.1.0

# faiss=" -I/data/fangyuec/faiss/faiss -L/data/fangyuec/faiss/faiss -lfaiss "
faiss_include="-I/data/fangyuec/faiss/faiss -I/data/fangyuec/faiss/build/"
# faiss_lib="-L/data/fangyuec/faiss/faiss -lfaiss -L/data/fangyuec/cuda-12.1/lib64"
faiss_lib="-I$HOME/local/include -L$HOME/local/lib -lfaiss -L/data/fangyuec/cuda-12.1/lib64"
# cuda=" -I/usr/local/cuda/include "
# lib_mkl="-Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_ilp64.a $MKLROOT/lib/intel64/libmkl_sequential.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"

lib_mkl="-Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_lp64.a $MKLROOT/lib/intel64/libmkl_sequential.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"

cuda_include=" "
lib="$faiss_include $cuda_include $lib_mkl"

# compiler="/home/fangyuec/local/cuda-12.1/bin/nvcc"
compiler="g++ -pthread -fopenmp -L/home/fangyuec/local/lib -lblas"

# $compiler -std=c++17 "$lib" -c -o SubGraphSearch.o SubGraphSearch.cpp $faiss_lib -o out
# $compiler -std=c++17 "$lib" -c -o Graph.o Graph.cpp $faiss_lib -o out
# $compiler -std=c++17 "$lib" -c -o Cache.o Cache.cpp $faiss_lib -o out
# $compiler -std=c++17 "$lib" -c -o Utils.o Utils.cpp $faiss_lib -o out
# $compiler -std=c++17 "$lib" -c -o CommonSubGraph.o CommonSubGraph.cpp $faiss_lib -o out
# $compiler -std=c++17 "$lib" -c -o TurboIso.o TurboIso.cpp $faiss_lib -o out

company="Cache.o Utils.o Graph.o SubGraphSearch.o CommonSubGraph.o TurboIso.o"

$compiler -std=c++17 "$lib" main.cpp $company $faiss_lib -o origin
# $compiler -std=c++17 "$lib" faiss1.cpp $company $faiss_lib -o faiss1
$compiler -std=c++17 "$lib" faiss_index_building.cpp $company $faiss_lib -o faiss_index_building
