import random
import sys

def generate_matrix(filename, rows, cols):
    # Open the file for writing
    with open(filename, 'w') as f:
        # Write matrix dimensions
        f.write(f"{rows} {cols}\n")
        # Generate matrix data and write to the file
        for _ in range(rows):
            row = [f"{random.uniform(-100.0, 100.0):.2f}" for _ in range(cols)]
            f.write(" ".join(row) + "\n")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 6:
        print("Usage: python generate.py <rowsA> <colsA> <colsB> <matrixA.txt> <matrixB.txt>")
        sys.exit(-1)

    # Read dimensions from command line arguments
    rowsA = int(sys.argv[1])
    colsA = int(sys.argv[2])
    colsB = int(sys.argv[3])
    matrixA_filename = sys.argv[4]
    matrixB_filename = sys.argv[5]

    # Validate dimensions
    if rowsA <= 0 or colsA <= 0 or colsB <= 0:
        print("Matrix dimensions must be positive integers.")
        sys.exit(-1)

    # Generate Matrix A (rowsA x colsA) and Matrix B (colsA x colsB)
    print(f"Generating Matrix A ({rowsA}x{colsA})...")
    generate_matrix(matrixA_filename, rowsA, colsA)

    print(f"Generating Matrix B ({colsA}x{colsB})...")
    generate_matrix(matrixB_filename, colsA, colsB)

    print(f"Matrix files '{matrixA_filename}' and '{matrixB_filename}' have been created.")
