#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "cuda.h"

// Run with nvcc -O3 -lcublas ../DIEKUHDA/kuhda.cu matmulttest.cu && ./a.out
#define THREADS 16

__global__ void fill_matrix(double* A, const int rows, const int cols) {
	int counter = 0;
	for (counter = 0; counter < rows*cols; ++counter) {
		A[counter] = cols;
	}
}

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, const int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

	double tmpSum = 0.0;
	int i;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
	C[ROW * N + COL] = tmpSum;
}

__global__ void gpu_mul(double * const A, double * const B, double * const C, const int rows_A, const int cols_A) {
	
	//Each Thread computes one element of C
	double C_element = 0.0;
	const int 	row = blockIdx.y * blockDim.y + threadIdx.y,
				col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < cols_A && col < cols_A) 
	{
		for (int k = 0; k < rows_A; ++k) {
			C_element += A[k * cols_A + row] * A[k * cols_A + col];
		}
		C[row * cols_A + col] = C_element;
	}
}


//extern void hello_device_wrapper(int printme);
//extern void gpu_mul_wrapper(double const * const A, double * const C, const int rows_A, const int cols_A);

//#include <cuda.h>
//#include "cuda_runtime.h"


int main() {			
	
	// Set cuda device
	gpuErrchk(cudaSetDevice(0));
	unsigned long n = 16384;

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

	gpuErrchk(cudaStreamSynchronize(copystream1));
	gpuErrchk(cudaStreamSynchronize(copystream2));

	// Perform the multiplications	
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaEventRecord(mainstart, mainstream));

	//fill_matrix<<<10, 10>>>(d_C->data, d_C->r, d_C->c);
	/*
	int N = n;
	dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512){
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
	}
	matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(d_A->data, d_B->data, d_C->data, n);
	*/

	// Grid dimmensions for multiplication
	// THREADS = 32 so sizeof(block) = 32*32 = 1024
	dim3 block(THREADS, THREADS);
	// grid = ceil(16384/32) = 512
	dim3 grid = dim3(ceil(((float)n)/block.x), ceil(((float)n)/block.y));
	
	// Perform the multiplications	
	gpu_mul<<<grid, block>>>(d_A->data, d_B->data, d_C->data, n, n);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaEventRecord(mainstop, mainstream));
    gpuErrchk(cudaEventSynchronize(mainstop));
    gpuErrchk(cudaEventElapsedTime(&mainstreamtimer, mainstart, mainstop));
	printf("Multiplication on device 0 took %lf seconds\n", mainstreamtimer/1000);
	
	//gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	TileGPUAddToHost(0, n, 0, n, d_C, h_C, copystream1);
	gpuErrchk(cudaStreamSynchronize(copystream1));

	// Test the result
	kuhdaTestM(0, n, 0, n, h_C);
	//kuhdaPrintM(h_C);
	
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
