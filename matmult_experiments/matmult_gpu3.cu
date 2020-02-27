#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cublas_v2.h>
#include "matmult_transfer_gpu.h"

extern "C" {

    // MULTI_GPU split A version - hiding overlap.
	void matmult_gpu3(int m, int n, int k, double *A_in, double *B, double *C_in)
	{
        omp_set_nested(1);
		long size_A = m * k * sizeof(double);
        long size_B = k * n * sizeof(double);
        long size_C = m * n * sizeof(double);

        int numDevices = 1;
        //cudaGetDeviceCount(&numDevices);
        m /= numDevices;
        size_A /= numDevices;
        size_C /= numDevices;

#pragma omp parallel for firstprivate(m, size_A, size_C)
        for (int device = 0; device < numDevices; device++) {

            double *A = A_in + m * k * device;
            double *C = C_in + m * n * device;

            // Allocate on device
            double *d_A, *d_B, *d_C;
            cudaSetDevice(device);
            allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);

            // Transfer B to device
            checkCudaErrors(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemset(d_C, 0, size_C));

            int numSplits = 8;
            m /= numSplits;
            size_A /= numSplits;
            size_C /= numSplits;
 
#pragma omp parallel for
            for (int split = 0; split < numSplits; split++) {
                cudaSetDevice(device);
                cublasHandle_t handle;
                cublasCreate(&handle);
                checkCudaErrors(cudaMemcpyAsync(d_A + m * k * split, A + m * k * split, size_A, cudaMemcpyHostToDevice));
                double time0 = omp_get_wtime();

                const double alpha = 1.0;
                const double beta = 0.0;
                cublasSetStream(handle, cudaStreamPerThread);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A + m * k * split, k, &beta, d_C + m * n * split, n); // Row major.
                checkCudaErrors(cudaMemcpyAsync(C + m * n * split, d_C + m * n * split, size_C, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
                time0 = omp_get_wtime() - time0; printf("Computing C = A * B           | %5.4f s %5.4f Gflops\n", time0, 2.0 * m * n * k * 1e-9 / time0);
                cublasDestroy(handle);
            }
            free_on_gpu(d_A, d_B, d_C);
        }
    }
}
