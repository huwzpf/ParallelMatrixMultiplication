#include <omp.h>
#include <time.h>

#include "utils.h"

double *matrix_multiply(double *A, double *B, int rowsA, int colsA, int colsB) {
    double *C = (double *)malloc(rowsA * colsB * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < rowsA * colsB; i++) {
        C[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }

    return C;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrixA.txt> <matrixB.txt> <result.txt>\n", argv[0]);
        return -1;
    }

    int rowsA, colsA, rowsB, colsB;
    double *A, *B, *C;

    A = read_double_matrix(argv[1], &rowsA, &colsA);
    B = read_double_matrix(argv[2], &rowsB, &colsB);

    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations with %d threads\n", omp_get_max_threads());
    omp_set_num_threads(omp_get_max_threads());

    float start_time = omp_get_wtime();
    C = matrix_multiply(A, B, rowsA, colsA, colsB);
    float end_time = omp_get_wtime();

    printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);

    write_double_matrix(argv[3], C, rowsA, colsB);

    free(A);
    free(B);
    free(C);

    return 0;
}
