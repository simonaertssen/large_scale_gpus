#include <stdio.h>
extern "C" {
#include "../DIEKUHDA/kuhda.h"
}

// Run with nvcc -O3 -arch=sm_70 -lcublas ../DIEKUHDA/kuhda.c kernels.cu matmulttest.cu
#define THREADS 32

// Inclusion of .cu in seperate file is necessary:
// See: https://stackoverflow.com/questions/30247592/compiling-and-linking-pure-c-and-cuda-code-warning-implicit-declaration-of-fun
#include "kernels.cuh"
#include "cuda.h"

//extern void hello_device_wrapper(int printme);
//extern void gpu_mul_wrapper(double const * const A, double * const C, const int rows_A, const int cols_A);

//#include <cuda.h>
//#include "cuda_runtime.h"


int main() {				
	unsigned long n = 1000, size = n * n * sizeof(double);
	unsigned long x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	// Containers for host and device matrices:
	matrix *h_A  = kuhdaMallocMP1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocMP1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocMP(n, n); // empty C matrix

	matrix *d_A  = kuhdaMallocDeviceM(n, n); 
	matrix *d_B  = kuhdaMallocDeviceM(n, n);
	matrix *d_C  = kuhdaMallocDeviceM(n, n);

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
	gpuErrchk(cudaEventRecord(mainstart, mainstream));

	// Send A and B to device 0...
	TileHostToGPU(0, n, 0, n, h_A, d_A, copystream1);
	TileHostToGPU(0, n, 0, n, h_B, d_B, copystream2);


	int testint = 12;
	hello_device_wrapper(testint);
	
	// Set cuda device
	gpuErrchk(cudaSetDevice(0));


	/*
	dim3 block(THREADS, THREADS);
	dim3 grid(ceil(((float)cols_A)/block.x), ceil(((float)rows_A)/block.y));
	*/

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Grid dimmensions for multiplication
	/*
	grid = dim3(ceil(((float)cols_C)/block.x), ceil(((float)rows_C)/block.y));
	
	// Perform the multiplications	
	gpuErrchk(cudaEventRecord(mainstart, mainstream));

	gpu_mul<<<grid, block>>>(d_A, d_C, rows_A, cols_A);
	*/

	gpuErrchk(cudaEventRecord(mainstop, mainstream));
    gpuErrchk(cudaEventSynchronize(mainstop));
    gpuErrchk(cudaEventElapsedTime(&mainstreamtimer, mainstart, mainstop));
	printf("Multiplication on device 0 took %lf seconds\n", mainstreamtimer/1000);
	
	//gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Test the result
    //kuhdaTestM(0, n, 0, n, h_C);
	
	// free all matrices
    printf("Cleaning up ..\n");
    gpuErrchk(cudaStreamDestroy(mainstream));
	gpuErrchk(cudaStreamDestroy(copystream1));
	gpuErrchk(cudaStreamDestroy(copystream2));
	
    gpuErrchk(cudaEventDestroy(mainstart));
	gpuErrchk(cudaEventDestroy(mainstop));

	kuhdaFreeM(h_A, 'p');
	kuhdaFreeM(h_B, 'p');
	kuhdaFreeM(h_C, 'p');
	kuhdaFreeM(d_A, 'c');
	kuhdaFreeM(d_B, 'c');
	kuhdaFreeM(d_C, 'c');
	return 0;
}
