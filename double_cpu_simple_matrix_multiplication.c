#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

double **read_matrix(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr,"Error opening file\n");
        exit(-1);
    }

    if (fscanf(file, "%d %d\n", rows, cols) != 2) {
        fprintf(stderr, "Invalid matrix format\n");
        fclose(file);
        exit(-1);
    }

    double **matrix = (double **)malloc((*rows) * sizeof(double *));
    for (int i = 0; i < *rows; i++) {
        matrix[i] = (double *)malloc((*cols) * sizeof(double));
    }

    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                fprintf(stderr, "Invalid matrix data\n");
                fclose(file);
                exit(-1);
            }
        }
    }

    fclose(file);
    return matrix;
}

void write_matrix(const char *filename, double **matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr,"Error opening file\n");
        exit(-1);
    }

    fprintf(file, "%d %d\n", rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%lf ", matrix[i][j]);
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

double **matrix_multiply(double **A, double **B, int rowsA, int colsA, int colsB) {
    double **C = (double **)malloc(rowsA * sizeof(double *));
    for (int i = 0; i < rowsA; i++) {
        C[i] = (double *)malloc(colsB * sizeof(double));
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0.0;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Function to free allocated matrix memory
void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrixA.txt> <matrixB.txt> <result.txt>\n", argv[0]);
        return -1;
    }

    int rowsA, colsA, rowsB, colsB;
    double **A, **B, **C;

    // Load matrices from files
    A = read_matrix(argv[1], &rowsA, &colsA);
    B = read_matrix(argv[2], &rowsB, &colsB);

    // Validate matrix dimensions for multiplication
    validate_dimensions(rowsA, colsA, rowsB, colsB);

    printf("Done loading data, starting computations with %d threads\n", omp_get_max_threads());
    omp_set_num_threads(omp_get_max_threads());
    // Measure the time of matrix multiplication
    double start_time = omp_get_wtime();
    C = matrix_multiply(A, B, rowsA, colsA, colsB);
    double end_time = omp_get_wtime();

    printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);

    // Save the result matrix to file
    write_matrix(argv[3], C, rowsA, colsB);

    // Free allocated memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}