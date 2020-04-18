// Program to test the allocation and sending of quarter tiles

#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas ../DIEKUHDA/kuhda.c tileTransferTest2.c && ./a.out


int main()
{
	unsigned long n = 8, size = n * n * sizeof(double);
	int x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	// Containers for host and device matrices
	matrix *h_A  = kuhdaMallocMdiag(n, n); // full size A matrix
	matrix *h_C  = kuhdaMallocM(n, n); // empty C matrix
	matrix *d_A1 = kuhdaMallocDeviceM(x, x); // tile of A on the device

	printf("sizeof d_A1 = %zu\n", sizeof(d_A1));

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	printf("n = %d\nThe full host matrix (h_A) is:\n", n);
	kuhdaPrintM(h_A);

	// Send the first quarter tile of A to device 0...
	gpuErrchk(cudaSetDevice(0));
	TileHostToGPU(0, x, 0, x, h_A, d_A1, stream);

	// printf("Copied tile on device:\n");
	// kuhdaPrintM(d_A1); // <- does not work for device matrices...

	// ...retrieve it again into C on the host
	TileGPUToHost(0, x, 0, x, d_A1, h_C, stream);

	printf("\nC matrix after first tile has been copied back from device:\n");
	kuhdaPrintM(h_C);

	cudaStreamDestroy(stream);

	// free all matrices
	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_C, 'k');
	kuhdaFreeM(d_A1, 'c');
	return 0;
}
