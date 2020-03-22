#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c dataTransfer.c && ./a.out

int main(){
	unsigned long n = 1000, size = n * n * sizeof(double);

	struct cudaDeviceProp prop;
	int i, device, devicecount;
	gpuErrchk(cudaGetDeviceCount(&devicecount));
	matrix *All[devicecount], *d_All[devicecount];;
	for (i = 0; i < devicecount; ++i){
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaGetDevice(&device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		printf("Allocating memory on %s: %d of %d\n", prop.name, device, devicecount);

		All[device] = kuhdaMallocMdiag(n, n);
		d_All[device] = kuhdaMatrixToGPU(n, n, All[device]);
	}

	// Send data around cyclicly:
	int destinations[4] = {1, 2, 3, 0}, destination;
	for (device = 0; device < devicecount; ++device){
		destination = destinations[device];
		gpuErrchk(cudaSetDevice(device));
		printf("Sending data from %d to %d\n", device, destination);

		gpuErrchk(cudaMemcpy(d_All[device]->data, d_All[destination]->data, n*n*sizeof(double), cudaMemcpyDeviceToDevice));
	}


	/*gpuErrchk(cudaMemcpy(d_B->data, d_A->data, n*n*sizeof(double), cudaMemcpyDeviceToDevice));
	kuhdaMatrixToHost(n, n, d_B, B);
	kuhdaPrintM(B); */
	for (i = 0; i < devicecount; ++i){
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaGetDevice(&device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		printf("Freeing memory on %s: %d of %d\n", prop.name, device, devicecount);

		kuhdaFreeM(All[device], 'k');
		kuhdaFreeM(d_All[device], 'c');
	}

	gpuErrchk(cudaDeviceReset());
	return 0;
}
