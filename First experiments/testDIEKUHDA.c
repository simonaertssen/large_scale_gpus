#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c testDIEKUHDA.c && ./a.out
// or better:
// make -f testDIEKUHDA_makefile

int main(){
	unsigned long n = 16000;
	matrix *A = kuhdaMallocMdiag(n, n);
	matrix *d_A = kuhdaMatrixToGPU(n, n, A);

	gpuErrchk(cudaSetDevice(0));

	int device, devicecount;
	gpuErrchk(cudaGetDevice(&device));
	gpuErrchk(cudaGetDeviceCount(&devicecount));
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("We are working on device %s: %d of %d\n", prop.name, device, devicecount - 1);

	long long gflops = kuhdaTimeDGEMM(d_A, 1, 1);

	kuhdaMatrixToHost(n, n, d_A, A);

	kuhdaFreeM(A, 'k');
	kuhdaFreeM(d_A, 'c');
	gpuErrchk(cudaDeviceReset());
	return 0;
}
