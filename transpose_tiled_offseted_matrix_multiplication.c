#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define BLOCK_SIZE 8

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

double *transpose_matrix(double *M, int rows, int cols) {
    double *M_T = (double *)malloc(rows * cols * sizeof(double));

    // Perform the transposition using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M_T[j * rows + i] = M[i * cols + j];
        }
    }

    return M_T;
}

double *matrix_multiply(double *A, double *B, int rowsA, int colsA, int colsB) {
    double *C = (double *)malloc(rowsA * colsB * sizeof(double));

    double *B_T = transpose_matrix(B, colsA, colsB);

    int num_threads = omp_get_max_threads();
    int columns_per_thread = (colsB + num_threads - 1) / num_threads; // For evenly spaced column start

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rowsA; i += BLOCK_SIZE) {
        int col = (omp_get_thread_num() * columns_per_thread) % colsB;
        for (int count = 0; count < (colsB + (colsB % BLOCK_SIZE)); count += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < rowsA; ii++) {
                for (int jj = col; jj < col + BLOCK_SIZE && jj < colsB; jj++) {
                    double sum = 0.0;
                    #pragma omp simd
                    for (int k = 0; k < colsA; k++) {
                        sum += A[ii * colsA + k] * B_T[jj * colsA + k];
                    }
                    C[ii * colsB + jj] = sum;
                }
            }

            // Move to the next column with wrap-around
            col += BLOCK_SIZE;
            if (col >= colsB) {
                col = 0;
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
    double *A, *B, *C;

    A = read_matrix(argv[1], &rowsA, &colsA);
    B = read_matrix(argv[2], &rowsB, &colsB);

    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations with %d threads\n", omp_get_max_threads());
    omp_set_num_threads(omp_get_max_threads());

    double start_time = omp_get_wtime();
    C = matrix_multiply(A, B, rowsA, colsA, colsB);
    double end_time = omp_get_wtime();

    printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);

    write_matrix(argv[3], C, rowsA, colsB);

    free(A);
    free(B);
    free(C);

    return 0;
}
