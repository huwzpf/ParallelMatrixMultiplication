import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

# Define matrix dimensions and file names
matrix_sizes = [
    (2000, 1500, 1000),
    (4000, 3000, 2000),
    (6000, 4000, 3000),
    (8000, 5000, 3000),
    (10000, 5000, 5000),
    (10000, 5000, 10000)
]

# Define source files for each executable
source_files = {
    "cuBLAS": "float_cublas.cu",
    "Simple": "float_cuda_simple_matrix_multiplication.cu",
    "Tiled": "float_cuda_tiled_matrix_multiplication.cu",
    "Optimized": "float_cuda_2d_tiled_register_cache_matrix_multiplication.cu",
}

# Define the resulting executables
executables = {
    name: f"./{name.lower()}_matrix_multiply" for name in source_files.keys()
}

# Regex pattern to extract the runtime from the output
runtime_pattern = r"Matrix multiplication completed in ([0-9.]+) seconds"

# Function to compile CUDA source files
def compile_executables():
    for name, source in source_files.items():
        print(f"Compiling {name}...")
        result = subprocess.run(
            ["nvcc", source, "-o", executables[name], "-lcudart", "-lcublas"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Compilation failed for {name}: {result.stderr}")
            raise RuntimeError(f"Failed to compile {source}")

# Function to generate matrices
def generate_matrices(rowsA, colsA, colsB, matrixA, matrixB):
    subprocess.run(["python3", "generate.py", str(rowsA), str(colsA), str(colsB), matrixA, matrixB])

# Function to measure runtime
def measure_runtime(executable, matrixA, matrixB, result_matrix, num_iterations=10):
    runtimes = []
    for _ in range(num_iterations):
        result = subprocess.run([executable, matrixA, matrixB, result_matrix],
                                capture_output=True, text=True)
        match = re.search(runtime_pattern, result.stdout)
        if match:
            runtimes.append(float(match.group(1)))
        else:
            print(f"Failed to extract runtime for {executable}: {result.stdout}")
    return np.mean(runtimes), (np.max(runtimes) - np.min(runtimes)) / 2 if runtimes else (None, None)

# Benchmarking
def benchmark():
    results = {key: [] for key in executables.keys()}
    total_operations = []

    for rowsA, colsA, colsB in matrix_sizes:
        ops = rowsA * colsA * colsB  # Total operations for matrix multiplication
        total_operations.append(ops)
        print(f"\nBenchmarking for {ops} operations ({rowsA}x{colsA} * {colsA}x{colsB})")
        matrixA, matrixB = "matrixA.txt", "matrixB.txt"
        result_matrix = "result_matrix.txt"

        # Generate matrices
        generate_matrices(rowsA, colsA, colsB, matrixA, matrixB)

        for name, executable in executables.items():
            print(f"Running {name}...")
            avg_runtime, uncertainty = measure_runtime(executable, matrixA, matrixB, result_matrix)
            if avg_runtime is not None:
                print(f"{name}: {avg_runtime:.6f} ± {uncertainty:.6f} seconds")
                results[name].append(avg_runtime)
            else:
                print(f"{name} failed to produce a valid runtime.")
                results[name].append(None)

    return results, total_operations

# Plotting
def plot_results(results, total_operations):
    plt.figure(figsize=(10, 6))

    for name, runtimes in results.items():
        if any(runtimes):
            plt.plot(total_operations, runtimes, label=name, marker='o')

    plt.xlabel("Total Operations (FLOPs)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Matrix Multiplication Runtimes vs Total Operations")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

# Main function
if __name__ == "__main__":
    try:
        compile_executables()
        results, total_operations = benchmark()
        plot_results(results, total_operations)
    except Exception as e:
        print(f"Error: {e}")
