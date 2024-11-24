#include <omp.h>
#include <time.h>

#include "utils.h"

#define BLOCK_SIZE 32

float *transpose_matrix(float *M, int rows, int cols) {
    float *M_T = (float *)malloc(rows * cols * sizeof(float));

    // Perform the transposition using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M_T[j * rows + i] = M[i * cols + j];
        }
    }

    return M_T;
}

float *matrix_multiply(float *A, float *B, int rowsA, int colsA, int colsB) {
    float *C = (float *)malloc(rowsA * colsB * sizeof(float));

    float *B_T = transpose_matrix(B, colsA, colsB);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i_tile = 0; i_tile < rowsA; i_tile += BLOCK_SIZE) {
        for (int j_tile = 0; j_tile < colsB; j_tile += BLOCK_SIZE) {
            for (int i = i_tile; i < i_tile + BLOCK_SIZE && i < rowsA; i++) {
                for (int j = j_tile; j < j_tile + BLOCK_SIZE && j < colsB; j++) {
                    float sum = 0.0;
                    for (int k = 0; k < colsA; k++) {
                        sum += A[i * colsA + k] * B_T[j * colsA + k];
                    }
                    C[i * colsB + j] = sum;
                }
            }
        }
    }

    free(B_T);
    return C;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrixA.txt> <matrixB.txt> <result.txt>\n", argv[0]);
        return -1;
    }

    int rowsA, colsA, rowsB, colsB;
    float *A, *B, *C;

    // Load matrices from files
    A = read_float_matrix(argv[1], &rowsA, &colsA);
    B = read_float_matrix(argv[2], &rowsB, &colsB);

    // Validate matrix dimensions for multiplication
    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations with %d threads\n", omp_get_max_threads());
    omp_set_num_threads(omp_get_max_threads());

    // Measure the time of matrix multiplication
    float start_time = omp_get_wtime();
    C = matrix_multiply(A, B, rowsA, colsA, colsB);
    float end_time = omp_get_wtime();

    printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);

    // Save the result matrix to file
    write_float_matrix(argv[3], C, rowsA, colsB);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}