// Program to test the allocation and sending of quarter tiles

#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas ../DIEKUHDA/kuhda.c tileTransferTest1.c && ./a.out

int main(){
	// Set verbose to 0 to mute output
	int verbose = 0;

	unsigned long n = 8, size = n * n * sizeof(double);
	int x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	printf("n = %d\n First tile expected output: %d x %d identity matrix.\n", n, x, x);

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocMdiag(n, n); // full size A matrix
	matrix *h_A1 = kuhdaMallocM(x, x);	  // first quarter tile

	// Send the first quarter tile of A to device 0...
	gpuErrchk(cudaSetDevice(0));
	matrix *d_A1 = kuhdaMatrixToGPU(x, x, h_A);

	// ...retrieve it again
	if (verbose == 1) printf("Transfer %d to h: ", 0);

	cudaStream_t *stream;
	stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
	gpuErrchk(cudaStreamCreate(&stream));

	// inputs: (void *dst, const void *src, size_t count, enum cudaMemcpyKind	kind, cudaStream_t stream)
	gpuErrchk(cudaMemcpy(h_A1->data, d_A1->data, sizex, cudaMemcpyDeviceToHost, stream));


	// print retrieved matrix (should be diagonal 4x4)
	kuhdaPrintM(h_A1);

}