#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cublas_v2.h>
#include "matmult_transfer_gpu.h"

extern "C" {

    // SINGLE_GPU split version - naive.
	void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C)
	{
        m = m/2;
        long elms = m * n;
		double *d_A, *d_B, *d_C;
        allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);
		transfer_to_gpu(m,n,k,A,B,C,d_A,d_B,d_C);
		double *d_A2, *d_B2, *d_C2;
        allocate_on_gpu(m, n, k, &d_A2, &d_B2, &d_C2);
		transfer_to_gpu(m,n,k,A+elms,B,C+elms,d_A2,d_B2,d_C2);
        printf("Computing C = A * B           ");
        double time0 = omp_get_wtime();
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaStream_t stream[2];
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasSetStream(handle, stream[0]);
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n); // Row major.
        cublasSetStream(handle, stream[1]);
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B2, n, d_A2, k, &beta, d_C2, n); // Row major.
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        cublasDestroy(handle);
        printf("| %5.4f s\n", omp_get_wtime() - time0);
        transfer_from_gpu(m, n, C, d_C);
        transfer_from_gpu(m, n, C + elms, d_C2);
        free_on_gpu(d_A, d_B, d_C);
	}
}
