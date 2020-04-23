// Program to test the allocation and sending of quarter tiles

#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas ../DIEKUHDA/kuhda.c tileTransferTest3.c && ./a.out

// What do we want to test: (in parallel)
// Send d_A1 and d_B1 to device 3 and d_A2 and d_B3 to device 2
// call kuhdamm() to try and compute in parallel

// TODO:
// write a test function

int main()
{
	unsigned long n = 8, size = n * n * sizeof(double);
	int x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	// Containers for host and device matrices
	matrix *h_A  = kuhdaMallocMP1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocMP1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocMP(n, n); // empty C matrix

	matrix *d_A1 = kuhdaMallocDeviceM(x, x); // upper left tile of A on the device
	matrix *d_B1 = kuhdaMallocDeviceM(x, x); // upper left tile of B on the device
	matrix *d_C1 = kuhdaMallocDeviceM(x, x); // upper left tile of B on the device

	// printf("sizeof d_A1 = %zu\n", sizeof(d_A1));

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	// printf("n = %d\nThe full host matrix (h_C) is:\n", n);
	// kuhdaPrintM(h_C);

	gpuErrchk(cudaSetDevice(0));
	// Send the first quarter tiles of A and B to device 0...
	TileHostToGPU(0, x, 0, x, h_A, d_A1, stream);
	TileHostToGPU(0, x, 0, x, h_B, d_B1, stream);

	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	gpuErrchk(cudaStreamSynchronize(stream));
	gpuErrchk(cudaEventRecord(start, stream));

	// Matrix multiplication: damm man that's fast 
	kuhdamm(d_A1, d_B1, d_C1, stream, 0);

	gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(stop, stream));
	gpuErrchk(cudaEventSynchronize(stop));

	float milliseconds = 0;
	gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("This took %lf seconds\n", milliseconds/1000);

	// ...retrieve it again into C on the host
	TileGPUToHost(0, x, 0, x, d_C1, h_C, stream);
	kuhdaTestM(0, x, 0, x, h_C);

	//printf("\nC matrix after first tile has been copied back from device:\n");
	//kuhdaPrintM(h_C);

	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(stop));
	cudaStreamDestroy(stream);


	// free all matrices
	kuhdaFreeM(h_A, 'p');
	kuhdaFreeM(h_B, 'p');
	kuhdaFreeM(h_C, 'p');
	kuhdaFreeM(d_A1, 'c');
	kuhdaFreeM(d_B1, 'c');
	kuhdaFreeM(d_C1, 'c');

	gpuErrchk(cudaDeviceReset());
	return 0;
}
