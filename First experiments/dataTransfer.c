#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c dataTransfer.c && ./a.out

int main(){
	unsigned long n = 10;

	gpuErrchk(cudaSetDevice(0));
	struct cudaDeviceProp prop;
	int i, device, devicecount;
	gpuErrchk(cudaGetDeviceCount(&devicecount));
	for (i = 0; i < devicecount; ++i){
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaGetDevice(&device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		printf("We are working on device %s: %d of %d\n", prop.name, device, devicecount - 1);
	}

	// Send over some data between devices:
	gpuErrchk(cudaSetDevice(0));
	gpuErrchk(cudaGetDevice(&device));
	gpuErrchk(cudaGetDeviceProperties(&prop, device));
	printf("We are working on device %s: %d of %d\n", prop.name, device, devicecount - 1);

	matrix *A = kuhdaMallocMdiag(n, n);
	matrix *d_A = kuhdaMatrixToGPU(n, n, A);

	// Device 2:
	gpuErrchk(cudaSetDevice(2));
	gpuErrchk(cudaGetDevice(&device));
	gpuErrchk(cudaGetDeviceProperties(&prop, device));
	printf("We are working on device %s: %d of %d\n", prop.name, device, devicecount - 1);

	matrix *B = kuhdaMallocM1(n, n);
	kuhdaPrintM(B);
	matrix *d_B = kuhdaMatrixToGPU(n, n, B);

	gpuErrchk(cudaMemcpy(d_B->data, d_A->data, n*n*sizeof(double), cudaMemcpyDeviceToDevice));
	kuhdaMatrixToHost(n, n, d_B, B);
	kuhdaPrintM(B);

	kuhdaFreeM(A, 'k');
	kuhdaFreeM(d_A, 'c');
	kuhdaFreeM(B, 'k');
	kuhdaFreeM(d_B, 'c');
	gpuErrchk(cudaDeviceReset());
	return 0;
}
