# Parallel Matrix Multiplication project for Parallel Programming course
## General description
This repository contains implementations of matrix multiplication algorithms for both GPU and CPU with different optimizations. 

## Performance results
CUDA implementation (`float_cuda_2d_tiled_register_cache_matrix_multiplication.cu`) is 17.3 % slower than `cublasSgemm()` from `cublas_v2.h` and 567 % faster than algorithm without any optimizations on large matrices. 

![Time plot](benchmark_results.png)

## TODO
- Refactor the code to use single main function and just include the kernels at compile time
- Add CPU performance comparison to some library implementation