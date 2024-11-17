#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16
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
    // 3. Each thread initializes local variable val that will accumulate parial results for it's corresponding output C element
    // 4. Offset along A columns (being the same as B rows) determining which submatrix is currently computed is initialized to 0
    // 5. Each thread fetches single submatrix element into corresponding cell of As,Bs
    // 6. Each thread does multiplication of As * Bs computing partial value for single element of C and adds the result to val
    // 7. Submatrix offset is incremented
    // 8. Steps 5-8 are repeated until pointer == colsA
    // 9. C elements are fully computed and are copied to global memory

    int subMatrixRow = threadIdx.y;
    int subMatrixCol = threadIdx.x;

    // Row along which given thread moves through A and column along which it moves through B are constant 
    int rowA = blockIdx.y * blockDim.y + threadIdx.y;
    int colB = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double As [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs [BLOCK_SIZE][BLOCK_SIZE];

    double val = 0;


    for (int offset = 0; offset < colsA; offset+=BLOCK_SIZE) { 
        // Padding the matrices and keeping the kernel without range checking actually makes it slower
        
        // Each thread moves through columns of matrix A
        int colA = offset + threadIdx.x;
        if (rowA < rowsA && colA < colsA) {
            As[subMatrixRow][subMatrixCol] = A[rowA * colsA + colA];
        }
        else {
            As[subMatrixRow][subMatrixCol] = 0.0;
        }

        // Each thread moves through rows of matrix B
        int rowB = offset + threadIdx.y;
        if (rowB < colsA && colB < colsB) {
            Bs[subMatrixRow][subMatrixCol] = B[rowB * colsB + colB];
        }
        else {
            Bs[subMatrixRow][subMatrixCol] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += As[subMatrixRow][i] * Bs[i][subMatrixCol];
        }

        __syncthreads();
    }

    if (rowA < rowsA && colB < colsB)
        C[rowA * colsB + colB] = val;
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

    dim3 blockDim(16, 16);
    dim3 gridDim(CEIL_DIVISION(colsB, blockDim.x), CEIL_DIVISION(rowsA, blockDim.y));

    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

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