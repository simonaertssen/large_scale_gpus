#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "math.h"
#include "cublas_v2.h"
#include <limits.h>

// PROBLEMS WITH SIZE OF MATRIX A, DUE TO INDICES I AND J
// Results from the latest session:
// n = 2*16384
// Time = 9.0686 seconds
// 775.978 GFLOPS


// This script contains some numerical tests to get to know cublas 
// and how to split up matrices in blocks for gpu computation.
// Run with:
// nvcc -lcublas -lgomp experiment_1.cu && ./a.out

void test(double * matrix, int dim){
    unsigned int i, j;
    double sum = 0.0;
    int count = 0;
    for (i = 0; i  < dim; ++i){
        for (j = 0; j  < dim; ++j){
            sum += matrix[i*dim + j];
            printf("Test: sum is %lf \n", sum);
            count++;
        }
    }
    printf("Test: count is %d, dim is %d, sum is %lf \n", count, dim*dim*dim, sum);
    }


int main(){
    // After running into trouble with large numbers, we need to check the max size
    // of the matrix dimensions.
    // printf("Max value of type int  = %d \n", INT_MAX);
    // printf("Max value of type uint = %zu \n", UINT_MAX);

    // Counters and timers:
    int i, j;
    double t1, t2, gflops;

    // Size of testmatrix = pow(2,14) = 16384
    int n = 16384*4;
    printf("n = %d\n", n);

    // Allocate a matrix A and C of size n*n
    double *A = (double*) malloc(n * n * sizeof(double));
    double *C = NULL;
    //cudaMalloc(&C, n * n * sizeof(double))
    if (A == NULL) printf("A is NULL\n");
    if (cudaMalloc(&C, n * n * sizeof(double)) != 0) printf("CudaMalloc failed\n");
    
    // Fill as ones
    for (i = 0; i  < n; ++i){
        for (j = 0; j  < n; ++j){
            //A[i*n + j] = 1.0;
            printf("i = %d, j = %d\n", i, j);
            *(A + i*n + j) = 1.0;
        }
    }
    printf("i = %zu, j = %zu\n", i, j);

    // Test if correct:
    //test(A, n);

    // Now the CUDA part:
    // Create cublas instance and stream
    cublasHandle_t handle;
    //cublasCreate(&handle)
    if ( cublasCreate(&handle) != 0 ) printf("cublasCreate faileds\n");


    cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
    //cudaStreamCreate(&stream[0]);
    if ( cudaStreamCreate(&stream[0]) != 0 ) printf("cudaStreamCreate faileds\n");

    // Send matrix to GPU:
    //cublasSetMatrix(n, n, sizeof(double), A, n, C, n);
    if ( cublasSetMatrix(n, n, sizeof(double), A, n, C, n) != 0 ) printf("cublasSetMatrix faileds\n");

    // Set matrix coefficients
    double alpha = 1.0;
    double beta  = 0.0;

    // Set CUDA stream
    //cublasSetStream(handle, stream[0]);
    if ( cublasSetStream(handle, stream[0]) != 0 ) printf("cublasSetStream faileds\n");

 
    // DGEMM: A = alpha*A*A + beta*A
    t1 = omp_get_wtime();
    cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    C, n,
                    C, n,
                    &beta,
                    C, n);

    cudaDeviceSynchronize();
    t2 = omp_get_wtime();

    // Time the computations as in "How to compute GFLOPS for GEMM BLAS?" - nvidia forum
    printf("Time: %lf \n", t2-t1);
    gflops = ((long int)n*(long int)n*(2*(long int)n + 2)) / (10e9 *(t2 - t1));
    //gflops = (2 * (long int)n * (long int)n * (long int)n ) / (10e9 *(t2 - t1));

    //printf("Compute: (%d*%d*(2*%d + 2)) / (10e9 *(%lf - %lf)) \n", n, n, n, t2, t1);
    //printf("Compute: %d \n", n*n*n);
    printf("Timed %lf GFLPS .. hah \n", gflops);

    //cublasGetMatrix(n, n, sizeof(double*), C, n, A, n);
    if ( cublasGetMatrix(n, n, sizeof(double*), C, n, A, n) != 0 ) printf("cublasGetMatrix faileds\n");

    
    // test(A,n);

    free(A);
    cudaFree(C);
    cublasDestroy(handle);

    // https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits.
    cudaDeviceReset();

    return 0;
}