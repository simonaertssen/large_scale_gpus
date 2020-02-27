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
	void matmult_gpu4(int m, int n, int k, double *A_in, double *B, double *C_in)
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

        m /= numDevices;
        size_A /= numDevices;
        size_C /= numDevices;

        //#pragma omp parallel for firstprivate(m, size_A, size_C)
        for (int device = 0; device < numDevices; device++) {

            cudaSetDevice(device);

            double *A = A_in + m * lda * device;
            double *C = C_in + m * ldc * device;

            cudaStream_t stream[numSplitsPerDevice];
            cudaEvent_t event[numSplitsPerDevice];
            for (int split_m = 0; split_m < numSplitsPerDevice; split_m++) {
                cudaStreamCreate(&stream[split_m]);
                cudaEventCreate(&event[split_m]);
            }

            // Allocate on device
            double *d_A, *d_B, *d_C;
            allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);
            //CHECK(cudaMemset(d_C, 2, size_C));

            m /= numSplitsPerDevice;
            size_A /= numSplitsPerDevice;
            size_C /= numSplitsPerDevice;
            n /= numSplits;
            size_B /= numSplits;

            cublasHandle_t handle;
            cublasCreate(&handle);

            //#pragma omp parallel for
            for (int split_m = 0; split_m < numSplits; split_m++) {

                //cudaSetDevice(device);
                CHECK(cudaMemcpyAsync(d_A + m * lda * split_m, A + m * lda * split_m, size_A, cudaMemcpyHostToDevice, stream[split_m]));
                CHECK(cudaMemcpy2DAsync(d_B + n * split_m, ldb * sizeof(double), B + n * split_m, ldb * sizeof(double), n * sizeof(double), k, cudaMemcpyHostToDevice, stream[split_m]));
                CHECK(cudaEventRecord(event[split_m], stream[split_m]));
            }

            for (int split_m = 0; split_m < numSplits; split_m++) {
                for (int split_n = 0; split_n < numSplits; split_n++) {

                    CHECK(cudaStreamWaitEvent(stream[split_m], event[split_n], 0));

                    double time0 = omp_get_wtime();
                    const double alpha = 1.0;
                    const double beta = 0.0;
                    cublasSetStream(handle, stream[split_m]);
                    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B + n * split_n, ldb, d_A + m * lda * split_m, lda, &beta, d_C + m * ldc * split_m + n * split_n, ldc);
                    CHECK(cudaMemcpy2DAsync(C + m * ldc * split_m + n * split_n, ldc * sizeof(double), d_C + m * ldc * split_m + n * split_n, ldc * sizeof(double), n * sizeof(double), m, cudaMemcpyDeviceToHost, stream[split_m]));
                    //printf("Computing C = A * B           | %5.4f s\n", omp_get_wtime() - time0);
                }
            }
            for (int split_m = 0; split_m < numSplits; split_m++) {
                CHECK(cudaStreamSynchronize(stream[split_m]));
            }
            cublasDestroy(handle);
            free_on_gpu(d_A, d_B, d_C);
        }
    }
}
