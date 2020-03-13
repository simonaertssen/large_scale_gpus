#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "math.h"
#include "cublas_v2.h"
#include <limits.h>

// This script contains some numerical tests to get to know cublas 
// and how to split up matrices in blocks for gpu computation.
// Run with:
// nvcc -lcublas -lgomp experiment_1.cu && ./a.out

void test(double * matrix, int dim){
    int i, j;
    long long sum = 0;
    int count = 0;
    for (i = 0; i < dim; ++i){
        for (j = 0; j < dim; ++j){
            sum += (long long) matrix[i*dim + j];
            count++;
        }
    }
    long long testdim = (long long)dim*(long long)dim*(long long)dim;
    if (testdim == sum){
        // This should be true because all matrices are filled with ones.
        printf("Test: count is %d, dim**3 is %d, sum is %.1ld \n", count, (long int)dim*(long int)dim*(long int)dim, sum);
    }
    }


int main(){
    // Counters and timers:
    int i, j, left, right;
    double t1, t2, gflops;

    // Size of testmatrix = pow(2,14) = 16384
    int n = 16384, sub_n = 25;
    int n_squared = n*n, sub_n_squared = sub_n*sub_n;

    // Allocate a matrix A and C of size n*n
    double *A = (double*) malloc(n_squared * sizeof(double));
    double *C = NULL;
    //cudaMalloc(&C, n_squared * sizeof(double))
    if (A == NULL) printf("A is NULL\n");
    
    if (cudaMalloc(&C, n_squared * sizeof(double)) != 0){
        fprintf(stderr, "CudaMalloc failed: matrix is of size %ldGB which is larger than 16GB (V100 memory).\n", n_squared * sizeof(double) / 10e9);
        exit(-1);
        }
    
    // Fill as ones
    for (i = 0; i < n; i++){
        for (j = 0; j  < n; j++){
            *(A + i*n + j) = 1.0;
        }
    }

    // Test if correct:
    //test(A, n);

    // Now the CUDA part:
    // Create cublas instance and stream
    cublasHandle_t handle;
    if ( cublasCreate(&handle) != 0 ) printf("cublasCreate faileds\n");
    cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
    if ( cudaStreamCreate(&stream[0]) != 0 ) printf("cudaStreamCreate faileds\n");

    // Set matrix coefficients
    double alpha = 1.0;
    double beta  = 0.0;

    // Set CUDA stream
    cublasSetStream(handle, stream[0]);
    if ( cublasSetStream(handle, stream[0]) != 0 ) printf("cublasSetStream failed\n");
 
    // DGEMM: A = alpha*A*A + beta*A
    t1 = omp_get_wtime();
    
    #pragma parallel for
    for (int i = 0; i < 4; ++i){
        left = sub_n*i, right = sub_n*(i+1);

        cudaSetDevice(i);

        // Send matrix to GPU:
        if ( cublasSetMatrix(n, n, sizeof(double), A, n, C, n) != 0 ) printf("cublasSetMatrix faileds\n");
    
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, C, n, C, n, &beta, C, n);

        // Return the matrix
        if ( cublasGetMatrix(n, n, sizeof(double*), C, n, A, n) != 0 ) printf("cublasGetMatrix faileds\n");

    }
    
    //cudaDeviceSynchronize();
    t2 = omp_get_wtime();

    // Time the computations as in "How to compute GFLOPS for GEMM BLAS?" - nvidia forum
    // See: https://devtalk.nvidia.com/default/topic/482834/how-to-compute-gflops-for-gemm-blas/
    printf("Elapsed itme is %lf seconds\n", t2 - t1);
    gflops = 4*(long long)(n_squared * (2*n + 2)) / (10e9 *(t2 - t1));
    printf("Timed %lf GFLPS .. hah \n", gflops);

    test(A,n);

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