#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cublas_v2.h>
#include "matmult_transfer_gpu.h"
#define CHECK checkCudaErrors

extern "C" {

    // SINGLE_GPU split A and B version - hiding overlap.
	void matmult_gpu5(int m, int n, int k, double *A_in, double *B, double *C_in)
	{
        omp_set_nested(1);

        int numSplits = 4;
        int numDevices = 1;
        //cudaGetDeviceCount(&numDevices);
        int numSplitsPerDevice = numSplits / numDevices;

        int lda = k;
        int ldb = n;
        int ldc = n;
        
		long size_A = m * lda * sizeof(double);
        long size_B = k * ldb * sizeof(double);
        long size_C = m * ldc * sizeof(double);

        size_A /= numDevices;
        size_C /= numDevices;

        //#pragma omp parallel for firstprivate(m, size_A, size_C)
        for (int device = 0; device < numDevices; device++) {

            cudaSetDevice(device);

            double *A = A_in + m * lda * device;
            double *C = C_in + m * ldc * device;

            cudaStream_t stream[numSplits][numSplitsPerDevice];
            cudaEvent_t event[numSplits][numSplitsPerDevice];
            for (int split_m = 0; split_m < numSplitsPerDevice; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {
                    cudaStreamCreate(&stream[split_m][split_n]);
                    cudaEventCreate(&event[split_m][split_n]);
                }
            }

            // Allocate on device
            double *d_A, *d_B, *d_C;
            allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);
            //CHECK(cudaMemset(d_C, 2, size_C));

            m /= numSplitsPerDevice;
            n /= numSplits;
            k /= numSplits;
            size_A /= numSplits * numSplitsPerDevice;
            size_C /= numSplits * numSplitsPerDevice;
            size_B /= numSplits * numSplits;

            cublasHandle_t handle;
            cublasCreate(&handle);

            for (int split_m = 0; split_m < numSplits; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {
                    CHECK(cudaMemcpy2DAsync(d_A + m * lda * split_m + n * split_n, lda * sizeof(double), A + m * lda * split_m + n * split_n, lda * sizeof(double), k * sizeof(double), m, cudaMemcpyHostToDevice, stream[split_m][split_n]));
                    CHECK(cudaMemcpy2DAsync(d_B + m * ldb * split_m + n * split_n, ldb * sizeof(double), B + m * ldb * split_m + n * split_n, ldb * sizeof(double), n * sizeof(double), k, cudaMemcpyHostToDevice, stream[split_m][split_n]));
                    CHECK(cudaEventRecord(event[split_m][split_n], stream[split_m][split_n]));
                }
            }
            for (int split_m = 0; split_m < numSplits; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {
                    for (int split_k = 0; split_k < numSplits; split_k++) {

                        CHECK(cudaStreamWaitEvent(stream[split_m][split_n], event[split_m][split_k], 0));

                        //double time0 = omp_get_wtime();
                        const double alpha = 1.0;
                        const double beta = 1.0;
                        cublasSetStream(handle, stream[split_m][split_n]);
                        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B + m * ldb * split_n + n * split_k, ldb, d_A + m * lda * split_m + n * split_n, lda, &beta, d_C + m * ldc * split_m + n * split_n, ldc);
                        //printf("Computing C = A * B           | %5.4f s\n", omp_get_wtime() - time0);
                    }
                }
            }
            for (int split_m = 0; split_m < numSplits; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {
                    CHECK(cudaMemcpy2DAsync(C + m * ldc * split_m + n * split_n, ldc * sizeof(double), d_C + m * ldc * split_m + n * split_n, ldc * sizeof(double), n * sizeof(double), m, cudaMemcpyDeviceToHost, stream[split_m][split_n]));
                }
            }

            for (int split_m = 0; split_m < numSplits; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {
                    CHECK(cudaStreamSynchronize(stream[split_m][split_n]));
                }
            }
            cublasDestroy(handle);
            free_on_gpu(d_A, d_B, d_C);
        }
    }
}
