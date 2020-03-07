#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h"
 
 // nvcc -g -O3 -arch=sm_70 -lcublas batchCublasTest.cu

int main(int argc, char* argv[])
{
    int i,j,k,index;
 
    // Linear dimension of matrices
    int dim = 4000;
 
    // Number of A,B,C matrix sets
    int batch_count = 2;
 
    // Allocate host storage for batch_count A,B,C square matrices
    double **A, **B, **C;
    A = (double**)malloc(batch_count*sizeof(double*));
    B = (double**)malloc(batch_count*sizeof(double*));
    C = (double**)malloc(batch_count*sizeof(double*));
    for(i=0; i<batch_count; i++) {
        A[i] = (double*)malloc(dim*dim*sizeof(double));
        B[i] = (double*)malloc(dim*dim*sizeof(double));
        C[i] = (double*)malloc(dim*dim*sizeof(double));
    }
 
    // Allocate device storage for batch_count A,B,C
    double **d_A, **d_B, **d_C;
    d_A = (double**)malloc(batch_count*sizeof(double*));
    d_B = (double**)malloc(batch_count*sizeof(double*));
    d_C = (double**)malloc(batch_count*sizeof(double*));
    for(i=0; i<batch_count; i++) {
        cudaMalloc((void**)&d_A[i], dim*dim*sizeof(double));
        cudaMalloc((void**)&d_B[i], dim*dim*sizeof(double));
        cudaMalloc((void**)&d_C[i], dim*dim*sizeof(double));
    }
    // Fill A,B diagonals with k*sin(i) data, C diagonal with k*cos(i)^2
    // Matrices are arranged column major
    for(k=0; k<batch_count; k++) {
        for(j=0; j<dim; j++) {
            for(i=0; i<dim; i++) {
                index = j*dim + i;
                if(i==j) {
                    (A[k])[index] = k*sin(index);
                    (B[k])[index] = sin(index);
                    (C[k])[index] = k*cos(index)*cos(index);
                }
        else {
                    (A[k])[index] = 0.0;
                    (B[k])[index] = 0.0;
                    (C[k])[index] = 0.0;
                }
            } // i   
        } // j
    } // k
 
    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Set input matrices on device
    for(i=0; i<batch_count; i++) {
        cublasSetMatrix(dim, dim, sizeof(double), A[i], dim, d_A[i], dim);
        cublasSetMatrix(dim, dim, sizeof(double), B[i], dim, d_B[i], dim);
        cublasSetMatrix(dim, dim, sizeof(double), C[i], dim, d_C[i], dim);
    }
 
    // Create a stream for every DGEMM operation
    cudaStream_t *streams = (cudaStream_t *) malloc(batch_count*sizeof(cudaStream_t));
    for(i=0; i<batch_count; i++)
        cudaStreamCreate(&streams[i]);
 
    // Set matrix coefficients
    double alpha = 1.0;
    double beta  = 1.0;
 
    // Launch each DGEMM operation in own CUDA stream
    for(i=0; i<batch_count; i++){
        // Set CUDA stream
        cublasSetStream(handle, streams[i]);
 
        // DGEMM: C = alpha*A*B + beta*C
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    dim, dim, dim,
                    &alpha,
                    d_A[i], dim,
                    d_B[i], dim,
                    &beta,
                    d_C[i], dim);
    }
 
    // Retrieve result matrix from device
    for(i=0; i<batch_count; i++) {
        cublasGetMatrix(dim, dim, sizeof(double), d_C[i], dim, C[i], dim);
    }
 
    // Simple sanity test, sum up all elements
    double sum = 0;
    for(k=0; k<batch_count; k++) {
        for(j=0; j<dim; j++) {
            for(i=0; i<dim; i++) {
                index = j*dim + i;
                sum += (C[k])[index];
            }
        }
    }
    printf("Element sum is: %f, should be: %d\n", sum, dim*(batch_count-1)*(batch_count)/2);   
 
    // Clean up resources
    for(i=0; i<batch_count; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
 
    return 0;
}
