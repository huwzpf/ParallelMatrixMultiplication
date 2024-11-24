#!/bin/bash

# List of run names (file names without the .c extension)
cpu_run_names=(
  "float_cpu_transpose_tiled_matrix_multiplication"
  "double_cpu_transpose_tiled_matrix_multiplication"
  "float_cpu_simple_matrix_multiplication"
  "double_cpu_simple_matrix_multiplication"
)

# Iterate over each run name
for run_name in "${cpu_run_names[@]}"; do
  printf "\n%s\n\n" "$run_name"
  gcc -fopenmp -O3 -o matrix_multiply "${run_name}.c" \
    && python3 measure.py \
    && python3 verify.py 1.txt 2.txt 3.txt "$run_name"
done

gpu_run_names=(
  "double_cuda_2d_tiled_register_cache_matrix_multiplication"
  "double_cuda_simple_matrix_multiplication"
  "double_cuda_tiled_matrix_multiplication"
  "float_cuda_2d_tiled_register_cache_matrix_multiplication"
  "float_cuda_simple_matrix_multiplication"
  "float_cuda_tiled_matrix_multiplication"
)

# Iterate over each run name
for run_name in "${gpu_run_names[@]}"; do
  printf "\n%s\n\n" "$run_name"
  nvcc "${run_name}.cu" -o matrix_multiply -lcudart \
    && python3 measure.py \
    && python3 verify.py 1.txt 2.txt 3.txt "$run_name"
done