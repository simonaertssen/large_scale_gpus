#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cublas_v2.h>
#include "matmult_transfer_gpu.h"

extern "C" {
	
	// V100 on input: [14336 14336 14336]
	// ## (16,16) : 4226973 MFLOPS
	void matmult_gpulib(int m,int n,int k,double *A,double *B,double *C)
	{
		double *d_A, *d_B, *d_C;
        allocate_on_gpu(m, n, k, &d_A, &d_B, &d_C);
		transfer_to_gpu(m,n,k,A,B,C,d_A,d_B,d_C);
        // printf("Computing C = A * B           ");
        double time0 = omp_get_wtime();
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n); // Row major.
		//cublasDgemm('N', 'N', m, n, k, 1.0, d_A, m, d_B, k, 0.0, d_C, m); // Col major.
		cudaDeviceSynchronize();
        cublasDestroy(handle);
        time0 = omp_get_wtime() - time0; 
        // printf("Computing C = A * B           | %5.4f s %5.4f Gflops\n", time0, 2.0 * m * n * k * 1e-9 / time0);
        transfer_from_gpu(m, n, C, d_C);
        free_on_gpu(d_A, d_B, d_C);
	}
}

