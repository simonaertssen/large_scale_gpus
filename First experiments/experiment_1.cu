#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h"

// This script contains some numerical tests to get to know cublas 
// and how to split up matrices in blocks for gpu computation.
// Run with:
// nvcc experiment_1.cu && ./a.out

void test(double * matrix, int dim){
    int i, j;
    double sum = 0.0;
    for (i = 0; i  < dim; ++i){
        for (j = 0; j  < dim; ++j){
            sum += matrix[i*dim + j];
        }
    }
    printf("Test: dim is %d, sum is %lf \n", dim, sum);
    }


int main(){
    // Counters:
    int i, j;

    // Size of testmatrix = pow(2,14) = 16384
    int n = 10;

    // Allocate a matrix A and C of size n*n
    double *A = (double*) malloc(n * n * sizeof(double));
    double *C = (double*) malloc(n * n * sizeof(double));
    cudaMalloc((void**)C, n * n * sizeof(double));
    
    // Fill as unit matrices
    for (i = 0; i  < n; ++i){
        for (j = 0; j  < n; ++j){
            A[i*n + j] = 0.0;
        }
    }
    for (i = 0; i < n*n; i+=n+1){
        A[i] = 1.0;
    } 

    // Test if correct:
    test(A, n);

    // Now the CUDA part:
    // Create cublas instance and stream
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
    cudaStreamCreate(&stream[0]);

    // Send matrix to GPU:
    cublasSetMatrix(n, n, sizeof(double*), A, n, C, n);

    // Set matrix coefficients
    double alpha = 1.0;
    double beta  = 0.0;

    // Set CUDA stream
    cublasSetStream(handle, stream[0]);
 
    // DGEMM: A = alpha*A*A + beta*A
    cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    C, n,
                    C, n,
                    &beta,
                    C, n);

    cublasGetMatrix(n, n, sizeof(double*), C, n, A, n);

    test(A,n);

    free(A);
    free(C);
    cudaFree(C);
    return 0;
}