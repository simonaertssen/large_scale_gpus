#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <omp.h>

extern "C" {

	inline void allocate_on_gpu(int m, int n, int k, double **d_A, double **d_B, double **d_C)
	{
		// Allocate matrices on device
		long size_A = m * k * sizeof(double);
		long size_B = k * n * sizeof(double);
		long size_C = m * n * sizeof(double);
        double time0 = omp_get_wtime();
		checkCudaErrors(cudaMalloc((void**)d_A, size_A)); 
		checkCudaErrors(cudaMalloc((void**)d_B, size_B)); 
		checkCudaErrors(cudaMalloc((void**)d_C, size_C)); 
        printf("Allocating on gpu             | %5.4f s\n", omp_get_wtime() - time0);
	}

	inline void transfer_to_gpu(int m, int n, int k, double *A, double *B, double *C, double *d_A, double *d_B, double *d_C)
	{
		long size_A = m * k * sizeof(double);
		long size_B = k * n * sizeof(double);
		long size_C = m * n * sizeof(double);

		// Transfer data to device
        double time0 = omp_get_wtime();       
		//checkCudaErrors(cudaMemcpyAsync(d_A, A, size_A, cudaMemcpyHostToDevice, cudaStreamPerThread));
		//checkCudaErrors(cudaMemcpyAsync(d_B, B, size_B, cudaMemcpyHostToDevice, cudaStreamPerThread));
		checkCudaErrors(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
        time0 = omp_get_wtime() - time0;
        printf("Transfering A and B to device | %5.4f s %3.2f GB %3.2f GB/s\n", time0, (double) (size_A + size_B) * 1e-9, (double) (size_A + size_B) * 1e-9 / time0);
		//checkCudaErrors(cudaMemsetAsync(d_C, 0, size_C, cudaStreamPerThread));
		checkCudaErrors(cudaMemset(d_C, 0, size_C));
	}

	inline void transfer_from_gpu(int m, int n, double *C, double *d_C)
	{
		// Transfer result to host
		long size_C = m * n * sizeof(double);
		checkCudaErrors(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));
	}

	inline void free_on_gpu(double *d_A, double *d_B, double *d_C)
	{
		// Clean up
		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
		checkCudaErrors(cudaFree(d_C));
    }
}
