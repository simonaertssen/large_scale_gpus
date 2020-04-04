// Program to test the allocation and sending of quarter tiles

#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas ../DIEKUHDA/kuhda.c tileTransferTest1.c && ./a.out

int main(){

	unsigned long n = 8, size = n * n * sizeof(double);
	int x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocMdiag(n, n); // full size A matrix
	matrix *h_C = kuhdaMallocM(n, n); // empty C matrix
	// matrix *d_A1 = kuhdaMallocDeviceM(x, x); // <- need this function!

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	printf("n = %d\nThe full matrix is:\n", n);
	kuhdaPrintM(h_A);

	// Send the first quarter tile of A to device 0...
	gpuErrchk(cudaSetDevice(0));
	TileHostToGPU(0, x, 0, x, h_A, d_A1, stream);

	// ...retrieve it again into C
	// TileGPUToHost(0, x, 0, x, d_A1, h_C, stream); // <- need this function!

	// print retrieved matrix
	printf("\nC matrix after first tile has been copied:\n");
	kuhdaPrintM(h_C);

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_C, 'k');
	kuhdaFreeM(d_A1, 'k');
	return 0;
}
