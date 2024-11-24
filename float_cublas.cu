#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

#include "utils.h"

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

    // Use cuBLAS for matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the matrix multiplication: C = alpha * A * B + beta * C
    // A: rowsA x colsA, B: colsA x colsB, C: rowsA x colsB
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                colsB, rowsA, colsA, 
                &alpha, 
                d_B, colsB, 
                d_A, colsA, 
                &beta, 
                d_C, colsB);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

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

    free(A);
    free(B);
    free(C);

    return 0;
}