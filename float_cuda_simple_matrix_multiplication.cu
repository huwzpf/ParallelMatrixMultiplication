#include <cuda_runtime.h>
#include <time.h>

#include "utils.h"

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        for (int k = 0; k < colsA; ++k) {
            C[row * colsB + col] += A[row * colsA + k] * B[k * colsB + col];
        }
    }
}

float *matrix_multiply_cuda(float *A, float *B, int rowsA, int colsA, int colsB) {
    float *C = (float *)malloc(rowsA * colsB * sizeof(float));

    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

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
    float *A, *B, *C;

    A = read_float_matrix(argv[1], &rowsA, &colsA);
    B = read_float_matrix(argv[2], &rowsB, &colsB);

    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations\n");

    float start_time = clock();
    C = matrix_multiply_cuda(A, B, rowsA, colsA, colsB);
    float end_time = clock();

    printf("Matrix multiplication completed in %f seconds\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);

    write_float_matrix(argv[3], C, rowsA, colsB);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}