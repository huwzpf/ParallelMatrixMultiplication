#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 64
// Divide x/y and round up
#define CEIL_DIVISION(x, y) ((x) + (y) - 1)/(y)

double *read_matrix(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        exit(-1);
    }

    if (fscanf(file, "%d %d\n", rows, cols) != 2) {
        fprintf(stderr, "Invalid matrix format\n");
        fclose(file);
        exit(-1);
    }

    double *matrix = (double *)malloc((*rows) * (*cols) * sizeof(double));

    for (int i = 0; i < (*rows) * (*cols); i++) {
        if (fscanf(file, "%lf", &matrix[i]) != 1) {
            fprintf(stderr, "Invalid matrix data\n");
            fclose(file);
            exit(-1);
        }
    }

    fclose(file);
    return matrix;
}

void write_matrix(const char *filename, double *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        exit(-1);
    }

    fprintf(file, "%d %d\n", rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%lf ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void validate_dimensions(int rowsA, int colsA, int rowsB, int colsB) {
    if (colsA != rowsB) {
        fprintf(stderr, "Matrix dimensions mismatch: %d != %d\n", colsA, rowsB);
        exit(-1);
    }
}

__global__ void matrixMultiplyKernel(double *A, double *B, double *C, int rowsA, int colsA, int colsB) {
    // Each block gets a square chunk (BLOCK_SIZE x BLOCK_SIZE) of C to compute based on chunks of A (BLOCK_SIZE x colsA) and B (colsA x BLOCK_SIZE)
    // It does so by dividing chunks of A and B into square submatrices and accumulating partial results of submatrix multiplication:
    // 1. Shared memory matrices As and Bs are allocated, each size of a BLOCK_SIZE x BLOCK_SIZE
    // 2. A columns (being the same as B rows) are divided into colsA/BLOCK_SIZE submatrices (assumption is they divide without remainder)
    // 3. Each thread initializes local variable c_values that will accumulate parial results for it's corresponding output C row
    // 4. Offset along A columns (being the same as B rows) determining which submatrix is currently computed is initialized to 0
    // 5. Each thread fetches row from matrix A into registers (a_values) memory and one row of B into shared memory
    // 6. Each thread does multiplication of a_values * Bs computing partial value for single row of C and adds the result to corresponding element of c_values
    // 7. Submatrix offset is incremented
    // 8. Steps 5-8 are repeated until pointer == colsA
    // 9. C elements are fully computed and are copied to global memory

    int subMatrixRow = threadIdx.y;
    // Divide grid into colsB / BLOCK_SIZE x rowsA / BLOCK_SIZE blocks
    // Each block calculates square piece of output of size BLOCK_SIZE x BLOCK_SIZE
    // Each block is divided into BLOCK_SIZE threads where each thread calculates one row of output matrix
    // Row A is index of row in output matrix (the same as row index in input matrix A)
    // startColB is index of first B column that will be used for calculations by given thread block
    int rowA = blockIdx.y * blockDim.y + threadIdx.y;
    int startColB = blockIdx.x * BLOCK_SIZE;

    // printf("Hello from thread %d, %d - rowA=%d, startColB=%d\n", threadIdx.x, threadIdx.y, rowA, startColB);
    double a_values [BLOCK_SIZE] = {0.0};
    __shared__ double Bs [BLOCK_SIZE * BLOCK_SIZE];
    double c_values [BLOCK_SIZE] = {0.0};

    for (int offset = 0; offset < colsA; offset+=BLOCK_SIZE) { 
        // Padding the matrices and keeping the kernel without range checking actually makes it slower
        
        // Each thread loads entire row from A submatrix
        if (rowA < rowsA) {
            for (int i = 0; i < BLOCK_SIZE; i++) {
                if (i + offset >= colsA) {
                    continue;
                }

                a_values[i] = A[rowA * colsA + i + offset];
                // printf("[Thread %d] a_values[%d] = (%d, %d) = %lf\n", threadIdx.y, i, rowA, i + offset, a_values[i]);
            }
        }

        // Each thread loads entire row from B submatrix
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if ((startColB + i) < colsB && (offset + subMatrixRow) < colsA) {
                Bs[subMatrixRow * BLOCK_SIZE + i] = B[(offset + subMatrixRow) * colsB + startColB + i];
                // printf("[Thread %d] Bs[%d, %d] = (%d, %d)\n", threadIdx.y, subMatrixRow, i, (offset + subMatrixRow), startColB + i);
            }
            else {
                Bs[subMatrixRow * BLOCK_SIZE + i] = 0;
            }
        }

        __syncthreads();

        // Compute each column of C
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < BLOCK_SIZE; col++) {
                //printf("[Thread %d] c_values[%d] += %lf * %lf\n", threadIdx.y, col, a_values[row], Bs[row * BLOCK_SIZE + col]);
                c_values[col] += a_values[row] * Bs[row * BLOCK_SIZE + col];
            }
        }

        __syncthreads();
    }

    if (rowA < rowsA) {
    // printf("RowA = %d\n", rowA);
        for (int col = 0; col < BLOCK_SIZE; col++) {
            if (col + startColB >= colsB) {
                continue;
            }
            // printf("startCol = %d row = %d col = %d c_values[%d] = %lf\n",rowA, col, col - startColB, c_values[col - startColB] );
            C[rowA * colsB + col + startColB] = c_values[col];
        }
    }
}

double *matrix_multiply_cuda(double *A, double *B, int rowsA, int colsA, int colsB) {
    double *C = (double *)malloc(rowsA * colsB * sizeof(double));

    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = colsA * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIVISION(colsB, BLOCK_SIZE), CEIL_DIVISION(rowsA, BLOCK_SIZE));    
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
    err =cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrixA.txt> <matrixB.txt> <result.txt>\n", argv[0]);
        return -1;
    }

    int rowsA, colsA, rowsB, colsB;
    double *A, *B, *C;

    A = read_matrix(argv[1], &rowsA, &colsA);
    B = read_matrix(argv[2], &rowsB, &colsB);

    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations\n");

    double start_time = clock();
    C = matrix_multiply_cuda(A, B, rowsA, colsA, colsB);
    double end_time = clock();

    printf("Matrix multiplication completed in %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    write_matrix(argv[3], C, rowsA, colsB);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}