#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 4
#define SHARED_CACHE_PER_THREAD_SIZE 2
#define SHARED_CACHE_SIZE SHARED_CACHE_PER_THREAD_SIZE * BLOCK_SIZE
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
    int startSubMatrixRow = threadIdx.y * SHARED_CACHE_PER_THREAD_SIZE;
    int startSubMatrixCol = threadIdx.x * SHARED_CACHE_PER_THREAD_SIZE;

    // Row along which given thread moves through A and column along which it moves through B are constant 
    int startRowA = blockIdx.y * blockDim.y * SHARED_CACHE_PER_THREAD_SIZE + startSubMatrixRow;
    int startColB = blockIdx.x * blockDim.x * SHARED_CACHE_PER_THREAD_SIZE + startSubMatrixCol;

    __shared__ double As [SHARED_CACHE_SIZE][SHARED_CACHE_SIZE];
    __shared__ double Bs [SHARED_CACHE_SIZE][SHARED_CACHE_SIZE];

    double acc[SHARED_CACHE_PER_THREAD_SIZE][SHARED_CACHE_PER_THREAD_SIZE] = {0};
    double a_vals[SHARED_CACHE_PER_THREAD_SIZE];
    double b_vals[SHARED_CACHE_PER_THREAD_SIZE];

    int currentRowA, currentColA, currentRowB, currentColB;


    // Each block loads a chunk of A and B matrices into shared memory
    // Each thread loads SHARED_CACHE_PER_THREAD_SIZE by SHARED_CACHE_PER_THREAD_SIZE subchunk
    // So entire chunk is a size of SHARED_CACHE_SIZE (BLOCK_SIZE * SHARED_CACHE_PER_THREAD_SIZE)
    for (int blockOffset = 0; blockOffset < colsA; blockOffset+=SHARED_CACHE_SIZE) { 
        // Padding the matrices and keeping the kernel without range checking actually makes it slower
        
        // Load all elements from A to As
        #pragma unroll
        for (int loadRowOffset = 0; loadRowOffset < SHARED_CACHE_PER_THREAD_SIZE; loadRowOffset++) {
            #pragma unroll
            for (int loadColOffset = 0; loadColOffset < SHARED_CACHE_PER_THREAD_SIZE; loadColOffset++) {
                currentRowA = startRowA + loadRowOffset;
                currentColA = blockOffset + startSubMatrixCol + loadColOffset;
                if (currentRowA < rowsA && currentColA < colsA) {
                    As[startSubMatrixRow + loadRowOffset][startSubMatrixCol + loadColOffset] = A[currentRowA * colsA + currentColA];
                }
                else {
                    As[startSubMatrixRow + loadRowOffset][startSubMatrixCol + loadColOffset] = 0.0;
                }
            }
        }

        // Load all elements from B to Bs
        #pragma unroll
        for (int loadRowOffset = 0; loadRowOffset < SHARED_CACHE_PER_THREAD_SIZE; loadRowOffset++) {
            #pragma unroll
            for (int loadColOffset = 0; loadColOffset < SHARED_CACHE_PER_THREAD_SIZE; loadColOffset++) {
                currentRowB = blockOffset + startSubMatrixRow + loadRowOffset;
                currentColB = startColB + loadColOffset;
                if (currentRowA < colsA && currentColB < colsB) {
                    Bs[startSubMatrixRow + loadRowOffset][startSubMatrixCol + loadColOffset] = A[currentRowB * colsB + currentColB];
                }
                else {
                    Bs[startSubMatrixRow + loadRowOffset][startSubMatrixCol + loadColOffset] = 0.0;
                }
            }
        }

        // Perform multiplication and accumulation
        #pragma unroll
        for (int sharedTileIndex = 0; sharedTileIndex < SHARED_CACHE_SIZE; ++sharedTileIndex) {

            // Load values from shared memory into registers
            #pragma unroll
            for (int subRow = 0; subRow < SHARED_CACHE_PER_THREAD_SIZE; ++subRow) {
                a_vals[subRow] = As[startSubMatrixRow + subRow][sharedTileIndex];
            }

            #pragma unroll
            for (int subCol = 0; subCol < SHARED_CACHE_PER_THREAD_SIZE; ++subCol) {
                b_vals[subCol] = Bs[sharedTileIndex][startSubMatrixCol + subCol];
            }

            // Compute products and accumulate in registers
            #pragma unroll
            for (int subRow = 0; subRow < SHARED_CACHE_PER_THREAD_SIZE; ++subRow) {
                #pragma unroll
                for (int subCol = 0; subCol < SHARED_CACHE_PER_THREAD_SIZE; ++subCol) {
                    acc[subRow][subCol] += a_vals[subRow] * b_vals[subCol];
                }
            }
        }
        __syncthreads();
    }


    // Write the accumulated values back to C
    #pragma unroll
    for (int subRow = 0; subRow < SHARED_CACHE_PER_THREAD_SIZE; ++subRow) {
        int globalRow = startRowA + subRow;
        printf("%d\n", globalRow);
        if (globalRow >= rowsA) continue;
        #pragma unroll
        for (int subCol = 0; subCol < SHARED_CACHE_PER_THREAD_SIZE; ++subCol) {
            int globalCol = startColB + subCol;
            if (globalCol >= colsB) continue;
            printf("%d %d\n", globalRow, globalCol);
            if (globalRow < 10 && globalCol < 10) {
                printf("%lf\n", acc[subRow][subCol]);
            }
            C[globalRow * colsB + globalCol] = acc[subRow][subCol];
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