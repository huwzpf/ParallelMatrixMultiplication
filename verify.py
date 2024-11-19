import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def read_matrix(filename):
    """Reads a matrix from a file and returns it as a NumPy array."""
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().strip().split())
        matrix = []
        for _ in range(rows):
            row = list(map(float, f.readline().strip().split()))
            matrix.append(row)
        return np.array(matrix)

def verify_multiplication(matrixA, matrixB, matrixC):
    """Verifies that matrixC is the product of matrixA and matrixB."""
    matrixA = matrixA.astype(np.float32)
    matrixB = matrixB.astype(np.float32)
    matrixC = matrixC.astype(np.float32)
    
    start_time = time.time()
    computed_result = np.dot(matrixA, matrixB)
    print(f"Verified in {time.time() - start_time:.4f} seconds")

    if np.allclose(computed_result, matrixC, atol=1):
        print("Verification successful: The result is correct.")
    else:
        print("Verification failed: The result is incorrect.")
        
        relative_diff = np.abs((computed_result - matrixC) / matrixC)
        max_relative_diff = np.max(relative_diff)
        avg_relative_diff = np.mean(relative_diff)
        percentiles = np.percentile(relative_diff, [25, 50, 75, 95, 99])
        
        total_entries = relative_diff.size
        greater_than_1_percent = np.sum(relative_diff > 0.01)
        percentage_greater_than_1_percent = (greater_than_1_percent / total_entries) * 100
        
        print(f"Maximum relative difference: {max_relative_diff:.6f}")
        print(f"Average relative difference: {avg_relative_diff:.6f}")
        print("Relative difference percentiles:")
        print(f"  25th: {percentiles[0]:.6f}")
        print(f"  50th: {percentiles[1]:.6f}")
        print(f"  75th: {percentiles[2]:.6f}")
        print(f"  95th: {percentiles[3]:.6f}")
        print(f"  99th: {percentiles[4]:.6f}")
        print(f"Percentage of entries with relative difference > 1%: {percentage_greater_than_1_percent:.6f}% = {greater_than_1_percent}/{(total_entries)}")
        
        plt.figure()
        plt.imshow(relative_diff, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Relative Difference")
        plt.title("Relative Differences Heatmap")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        heatmap_filename = "relative_differences_heatmap.png"
        plt.savefig(heatmap_filename)
        print(f"Heatmap saved as {heatmap_filename}")
        plt.close()
        
        plt.figure()
        plt.hist(relative_diff.flatten(), bins=50, color='blue', alpha=0.7, log=True)
        plt.title("Distribution of Relative Differences (Log Scale)")
        plt.xlabel("Relative Difference")
        plt.ylabel("Frequency (Log Scale)")
        histogram_filename = "relative_differences_histogram.png"
        plt.savefig(histogram_filename)
        print(f"Histogram saved as {histogram_filename}")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_result.py <matrixA.txt> <matrixB.txt> <result.txt>")
        sys.exit(-1)

    matrixA_filename = sys.argv[1]
    matrixB_filename = sys.argv[2]
    result_filename = sys.argv[3]

    print("Loading matrices...")
    matrixA = read_matrix(matrixA_filename)
    matrixB = read_matrix(matrixB_filename)
    matrixC = read_matrix(result_filename)

    print("Verifying the result...")
    verify_multiplication(matrixA, matrixB, matrixC)
