#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cublas_v2.h>
#include "matmult_transfer_gpu.h"

extern "C" {

    // SINGLE_GPU split version - hiding overlap.
	void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C)
	{
        cudaSetDevice(0);

		// Allocate on device
        double *d_A, *d_B, *d_C;
        allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);
		long size_B = k * n * sizeof(double);
		long size_C = m * n * sizeof(double);

		// Transfer B to device
		checkCudaErrors(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_C, 0, size_C));

        int numSplits = 8;
        m /= numSplits;
        long elms = m * n;
		long size_A = m * k * sizeof(double);
        size_C /= numSplits;
 
#pragma omp parallel for
        for (int split = 0; split < numSplits; split++) {

            cudaSetDevice(0);
            cublasHandle_t handle;
            cublasCreate(&handle);
            checkCudaErrors(cudaMemcpyAsync(d_A + m * k * split, A + m * k * split, size_A, cudaMemcpyHostToDevice));
            double time0 = omp_get_wtime();

            const double alpha = 1.0;
            const double beta = 0.0;
            cublasSetStream(handle, cudaStreamPerThread);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A + m * k * split, k, &beta, d_C + elms * split, n); // Row major.
            checkCudaErrors(cudaMemcpyAsync(C + elms * split, d_C + elms * split, size_C, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            cudaStreamSynchronize(cudaStreamPerThread);
            time0 = omp_get_wtime() - time0; printf("Computing C = A * B           | %5.4f s %5.4f Gflops\n", time0, 2.0 * m * n * k * 1e-9 / time0);

            cublasDestroy(handle);
        }

        free_on_gpu(d_A, d_B, d_C);
	}
}
