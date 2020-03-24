#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include <omp.h>

// Run with:
// nvcc -lgomp ../DIEKUHDA/kuhda.c dataTransferTiming.c && ./a.out

int main(){
	unsigned long n = 10000, size = n * n * sizeof(double);
	printf("Size = %u\n", size);

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
	double t1, t2, result;
	float milliseconds;
	//unsigned long result;
	cudaEvent_t start, stop;

	int destinations[4] = {1, 2, 3, 0}, destination, reps = 1000, rep;
	for (device = 0; device < devicecount; ++device){
		destination = destinations[device];
		gpuErrchk(cudaSetDevice(device));
		printf("Sending data from %d to %d\n", device, destination);

		gpuErrchk(cudaEventCreate(&start));
		gpuErrchk(cudaEventCreate(&stop));

		gpuErrchk(cudaEventRecord(start, 0));
		for (rep = 0; rep < reps; ++rep){
			gpuErrchk(cudaMemcpy((d_All[device])->data, (d_All[destination])->data, size, cudaMemcpyDeviceToDevice));
		}
		gpuErrchk(cudaDeviceSynchronize(device));

	   gpuErrchk(cudaEventRecord(stop, 0));
		gpuErrchk(cudaEventSynchronize(stop));
		gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

		result = (reps * size) / (1.0e6 * milliseconds ) ;
		printf("Registered a transfer of %.3lf Gb/s \n", result);

		gpuErrchk(cudaEventDestroy(start));
		gpuErrchk(cudaEventDestroy(stop));
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
