import numpy as np
import sys
import time

def read_matrix(filename):
    """Reads a matrix from a file and returns it as a NumPy array."""
    with open(filename, 'r') as f:
        # Read matrix dimensions
        rows, cols = map(int, f.readline().strip().split())
        # Initialize matrix as a list of lists
        matrix = []
        for _ in range(rows):
            row = list(map(float, f.readline().strip().split()))
            matrix.append(row)
        # Convert list of lists to NumPy array
        return np.array(matrix)

def verify_multiplication(matrixA, matrixB, matrixC):
    """Verifies that matrixC is the product of matrixA and matrixB."""
    # Compute the matrix product using NumPy

    start_time = time.time()
    computed_result = np.dot(matrixA, matrixB)
    print(f"Verified in {time.time() - start_time} seconds")

    # Check if the result matches the provided matrixC
    if np.allclose(computed_result, matrixC, atol=1e-6):
        print("Verification successful: The result is correct.")
    else:
        print("Verification failed: The result is incorrect.")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python verify_result.py <matrixA.txt> <matrixB.txt> <result.txt>")
        sys.exit(-1)

    # Read input filenames
    matrixA_filename = sys.argv[1]
    matrixB_filename = sys.argv[2]
    result_filename = sys.argv[3]

    # Load matrices from files
    print("Loading matrices...")
    matrixA = read_matrix(matrixA_filename)
    matrixB = read_matrix(matrixB_filename)
    matrixC = read_matrix(result_filename)

    # Verify the multiplication result
    print("Verifying the result...")
    verify_multiplication(matrixA, matrixB, matrixC)
