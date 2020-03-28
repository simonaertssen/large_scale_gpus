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
	matrix *h_A1 = kuhdaMallocM(x, x);	  // first quarter tile

	// Send the first quarter tile of A to device 0...
	gpuErrchk(cudaSetDevice(0));
	double *d_A1 = kuhdaTileToGPU(0, x, 0, x, h_A);

	printf("n = %d\nThe full matrix is:\n", n);
	kuhdaPrintM(h_A);

	// ...retrieve it again
	kuhdaTileToHost(x, x, d_A1, h_A1);

	// print retrieved matrix
	printf("\nFirst tile expected output: %d x %d identity matrix.\n", x, x);
	kuhdaPrintM(h_A1);

}
