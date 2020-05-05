#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "cuda.h"
#include <cublasXt.h>

// Run with nvcc -O3 -lcublas ../DIEKUHDA/kuhda.cu matmulttest.cu && ./a.out

int main() {			
    
    // Find GPU info
    int device_count, status = 0;
    gpuErrchk(cudaGetDeviceCount(&device_count));

	unsigned long n = 16384*2, m = n, k = n;

    // Containers for host and device matrices:
    printf("Allocating devices\n");
	matrix *h_A  = kuhdaMallocMP1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocMP1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocMP(n, n); // empty C matrix

	// matrix *d_A  = kuhdaMallocDeviceM(n, n); 
	// matrix *d_B  = kuhdaMallocDeviceM(n, n);
	// matrix *d_C  = kuhdaMallocDeviceM(n, n);

	// Make streams and copy data:
	cudaStream_t mainstream, copystream1, copystream2;
	gpuErrchk(cudaStreamCreate(&mainstream));
	gpuErrchk(cudaStreamCreate(&copystream1));
	gpuErrchk(cudaStreamCreate(&copystream2));

	// Allocate the timer:
    cudaEvent_t mainstart, mainstop;
	float mainstreamtimer;
    gpuErrchk(cudaEventCreate(&mainstart));
	gpuErrchk(cudaEventCreate(&mainstop));

	// Send A and B to device 0...
	//TileHostToGPU(0, n, 0, n, h_A, d_A, copystream1);
	//TileHostToGPU(0, n, 0, n, h_B, d_B, copystream2);

	gpuErrchk(cudaStreamSynchronize(copystream1));
	gpuErrchk(cudaStreamSynchronize(copystream2));

    // Perform the multiplications with CublasXt
    cublasXtHandle_t Xt_handle;
    status = cublasXtCreate(&Xt_handle);
    if (status != 0) {
        fprintf(stderr, "!!!! CUBLASXT handle creation error\n");
        return -1;
		}
	
    int devices[4] = {0, 1, 2, 3};
    status = cublasXtDeviceSelect(Xt_handle, device_count, devices);
    if (status != 0) {
        fprintf(stderr, "!!!! CUBLASXT device selection error\n");
        return -1;
        }
	status = cublasXtSetBlockDim(Xt_handle, 32);
	if (status != 0) {
        fprintf(stderr, "!!!! CUBLASXT block dim selection error\n");
        return -1;
        }
	
	double alpha = 1.0, beta  = 0.0;
    gpuErrchk(cudaEventRecord(mainstart, mainstream));
    status = cublasXtDgemm(Xt_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        h_A->data, m, h_B->data, k, &beta, h_C->data, m);
    if (status != 0) {
        fprintf(stderr, "!!!! CUBLASXT DGEMM performation error\n");
        return -1;
        }
    
    // Wait for multiplications to finish	
	gpuErrchk(cudaEventRecord(mainstop, mainstream));
    gpuErrchk(cudaEventSynchronize(mainstop));
    gpuErrchk(cudaEventElapsedTime(&mainstreamtimer, mainstart, mainstop));
	printf("Multiplication took %lf seconds\n", mainstreamtimer/1000);
	long int numerator    = (long int)(m * n) * (2 * ((long long)k) + 2);
	long long denominator = 1.0e6 * mainstreamtimer;
	long long gflops = numerator / denominator;
	printf("%lu GFLPS\n", gflops);
	
	gpuErrchk(cudaDeviceSynchronize());
	
	//TileGPUAddToHost(0, n, 0, n, d_C, h_C, copystream1);
	//gpuErrchk(cudaStreamSynchronize(copystream1));

	// Test the result
	kuhdaTestM(0, n, 0, n, h_C);
	//kuhdaPrintM(h_C);
	
	// free all matrices
    printf("Cleaning up ..\n");

    cublasXtDestroy(Xt_handle);

    gpuErrchk(cudaStreamDestroy(mainstream));
	gpuErrchk(cudaStreamDestroy(copystream1));
	gpuErrchk(cudaStreamDestroy(copystream2));
	
    gpuErrchk(cudaEventDestroy(mainstart));
	gpuErrchk(cudaEventDestroy(mainstop));

	kuhdaFreeM(h_A, 'p');
	kuhdaFreeM(h_B, 'p');
	kuhdaFreeM(h_C, 'p');
	// kuhdaFreeM(d_A, 'c');
	// kuhdaFreeM(d_B, 'c');
	// kuhdaFreeM(d_C, 'c');
	return 0;
}
