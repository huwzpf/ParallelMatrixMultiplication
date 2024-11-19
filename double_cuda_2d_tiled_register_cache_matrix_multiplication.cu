#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

// #define DEBUG
#define BLOCK_SIZE 128
#define BLOCK_STEP 8
#define TILE_SIZE 8

// Calculate row within submatrix by dividing linear index by number of columns
#define ROW_IN_SUBMATRIX(index, columns) (index / columns)
// Calculate column within submatrix by taking remainder of division of linear index by number of columns
#define COL_IN_SUBMATRIX(index, columns) (index % columns)

#ifdef DEBUG
    #define KERNEL_DEBUG(fmt, ...) \
        printf("[Thread %d %d (Block %d %d)] " fmt, \
               threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
#else
    #define KERNEL_DEBUG(fmt, ...);
#endif

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
    // Add some asserts just to make sure configuration isn't screwed up
    assert(BLOCK_SIZE % TILE_SIZE == 0);
    assert(BLOCK_STEP % TILE_SIZE == 0);
    int cElementsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    int cElementsPerThread = TILE_SIZE * TILE_SIZE;
    int numThreads = blockDim.x;
    assert(numThreads == (cElementsPerBlock / cElementsPerThread));

    int submatrixElementsPerThread =  (BLOCK_STEP * TILE_SIZE * TILE_SIZE)/(BLOCK_SIZE);
    int threadId = threadIdx.x;

    // Each thread loads submatrix elements from threadIndexStart to threadIndexStart + submatrixElementsPerThread
    int threadIndexStart = threadId * submatrixElementsPerThread;

    // Number of tiles per A row (or B column) in the block's submatrix
    int tilesPerRow = BLOCK_SIZE / TILE_SIZE;

    // Thread's tile position within the block
    int tileRowStart = TILE_SIZE * ROW_IN_SUBMATRIX(threadId, tilesPerRow);
    int tileColStart = TILE_SIZE * COL_IN_SUBMATRIX(threadId, tilesPerRow);
    
    // Each block gets a square chunk (BLOCK_SIZE x BLOCK_SIZE) of C to compute based on chunks of A (BLOCK_SIZE x colsA) and B (colsA x BLOCK_SIZE)
    // Each thread gets a TILE_SIZE x TILE_SIZE piece of C submatrix
    // It does so by dividing chunks of A and B into rectangular submatrices (As and Bs) and accumulating partial results of submatrix multiplication
    // Algorithm steps:
    // 1. Shared memory matrices As (BLOCK_SIZE x BLOCK_STEP) and Bs (BLOCK_STEP x BLOCK_SIZE) are allocated
    // 2. A columns (being the same as B rows) are divided into colsA/BLOCK_STEP submatrices
    // 3. Step offset along A columns (being the same as B rows) determining which submatrix is currently computed is initialized to 0
    // 4. Each thread initializes local variable results that will accumulate parial results for it's corresponding output C elements
    // 5. Each thread fetches range of submatrix elements into corresponding cell of As,Bs
    //    There are cElementsPerBlock / cElementsPerThread = (BLOCK_SIZE * BLOCK_SIZE)/(TILE_SIZE * TILE_SIZE) threads in block
    //    and size of As or Bs is BLOCK_SIZE * BLOCK_STEP, so each thread needs to load
    //    (BLOCK_SIZE * BLOCK_STEP) / ((BLOCK_SIZE * BLOCK_SIZE)/(TILE_SIZE * TILE_SIZE)) = (BLOCK_STEP * TILE_SIZE * TILE_SIZE)/(BLOCK_SIZE) elements
    // 6. Each thread computes it's tile by multiplying chunk of As (At of size TILE_SIZE x BLOCK_STEP) by chunk of Bs (Bt of size BLOCK_STEP x TILE_SIZE)
    //    It can be speed up by iteratively loading one column from At and one row from Bt to registers
    //    Then computing all partial results for tile that require this row + column pair and going to another pair
    //    tC[x,y] = sum for i from 0 to BLOCK_STEP (At[x, i] * Bt[i, y]),
    //    so it's more efficient to iterate from 0 to BLOCK_STEP and at each iteration compute partial result for each tC[x,y]
    // 7. Submatrix step offset is incremented
    // 8. Steps 5-8 are repeated until step offset is equal to colsA
    // 9. C elements are fully computed and are copied to global memory
     
    __shared__ double As[BLOCK_SIZE * BLOCK_STEP];
    __shared__ double Bs[BLOCK_STEP * BLOCK_SIZE];
    double results[TILE_SIZE * TILE_SIZE] = {0.0}; 

    double aCol[TILE_SIZE ] = {0.0}; 
    double bRow[TILE_SIZE] = {0.0}; 

    
    // Iterate over submatrices along A columns / B rows
    for (int offset = 0; offset < colsA; offset+=BLOCK_STEP) {
        // Load As elements with bound checking
        // If not within A bounds, set element to 0

        for (int i = threadIndexStart; i < threadIndexStart + submatrixElementsPerThread; i++) {
            int asRow = ROW_IN_SUBMATRIX(i, BLOCK_STEP);
            int asCol = COL_IN_SUBMATRIX(i, BLOCK_STEP); 

            int globalRow = blockIdx.y * BLOCK_SIZE + asRow; 
            int globalCol = offset + asCol;

            if (globalRow < rowsA && globalCol < colsA) {
                As[asRow * BLOCK_STEP + asCol] = A[globalRow * colsA + globalCol];
            } else {
                As[asRow * BLOCK_STEP + asCol] = 0.0; 
            }
        }
    
        // Load Bs elements with bound checking
        // If not within B bounds, set element to 0
        for (int i = threadIndexStart; i < threadIndexStart + submatrixElementsPerThread; i++) {
            int bsRow = ROW_IN_SUBMATRIX(i, BLOCK_SIZE);
            int bsCol = COL_IN_SUBMATRIX(i, BLOCK_SIZE); 

            int globalRow = offset + bsRow;
            int globalCol = blockIdx.x * BLOCK_SIZE + bsCol;

            if (globalRow < colsA && globalCol < colsB) {
                Bs[bsRow * BLOCK_SIZE + bsCol] = B[globalRow * colsB + globalCol];
            } else {
                Bs[bsRow * BLOCK_SIZE + bsCol] = 0.0;
            }
        }

        __syncthreads();

        // Iterate over the shared dimension of As and Bs (BLOCK_STEP)
        for (int blockOffset = 0; blockOffset < BLOCK_STEP; blockOffset++) {
            // Each thread loads TILE_SIZE elements from As into aCol
            for (int tileRowOffset = 0; tileRowOffset < TILE_SIZE; tileRowOffset++) {
                aCol[tileRowOffset] = As[(tileRowStart + tileRowOffset) * BLOCK_STEP + blockOffset];
            }
    
            // Each thread loads TILE_SIZE elements from Bs into bRow
            for (int tileColOffset = 0; tileColOffset < TILE_SIZE; tileColOffset++) {
                bRow[tileColOffset] = Bs[blockOffset * BLOCK_SIZE + (tileColStart + tileColOffset)];
            }
    
            // Compute partial results for tile
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    // Multiply the loaded column and row, accumulate the results
                    results[i * TILE_SIZE + j] += aCol[i] * bRow[j];
                }
            }
        }

        __syncthreads();
    }

    // Write tile results to C with bound checking
    for (int tileColOffset = 0; tileColOffset < TILE_SIZE; tileColOffset++) {
        for (int tileRowOffset = 0; tileRowOffset < TILE_SIZE; tileRowOffset++) {
            int globalRow = blockIdx.y * BLOCK_SIZE + tileRowStart + tileRowOffset;
            int globalCol = blockIdx.x * BLOCK_SIZE + tileColStart + tileColOffset;
            if (globalCol < colsB && globalRow < rowsA) {
                C[globalRow * colsB + globalCol] = results[tileRowOffset * TILE_SIZE + tileColOffset];
            }
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

    dim3 blockDim((BLOCK_SIZE * BLOCK_SIZE) / (TILE_SIZE * TILE_SIZE));
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

    printf("Matrix multiplication completed in %lf seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    write_matrix(argv[3], C, rowsA, colsB);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}