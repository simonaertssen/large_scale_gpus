#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include <omp.h>

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c dataTransfer.c && ./a.out

int main(){
	unsigned long n = 20000, size = n * n * sizeof(double);
	printf("This scripts times sending %u GB of memory between host and devices\n", 1.0e-9 * size);

	// Get the overall structure setup
	struct cudaDeviceProp prop;
	int device, devicecount;
	gpuErrchk(cudaGetDeviceCount(&devicecount));

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocMdiag(n, n), *d_All[devicecount];

	int i,j;
	// See if we do get and set the right devices:
	for (i = 0; i < devicecount; ++i){
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaGetDevice(&device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		printf("Allocating memory on %s: %d of %d\n", prop.name, device, devicecount);

		d_All[device] = kuhdaMatrixToGPU(n, n, h_A);

		// Make device able to send and receive data from peers:
		for (j = 0; j < devicecount; ++j){
			if (j == i) continue;
			gpuErrchk(cudaDeviceEnablePeerAccess(j, 0));
		}

	}

	// Time connections between devices:
	float milliseconds, results[16];
	int res_ctr = 0;
	cudaEvent_t start, stop;
	cudaStream_t *stream;

	// Now the device - device loop
	int src, dst, reps = 1000, rep, *accessible;
	for (src = 0; src < devicecount; ++src){
		// Set the current device:
		gpuErrchk(cudaSetDevice(src));

		for (dst = 0; dst < devicecount; ++dst){
			// Check if not on the diagonal:
			if (src == dst) continue;
			// Check if access is set:
			gpuErrchk(cudaDeviceCanAccessPeer(&accessible, dst, src));
			if (accessible == 0){
				printf("Device %d cannot access device %d", dst, src);
				continue;
			}
			printf("Transfer %d to %d: ", dst, src);

			// Now continue:
			// Create the stream for the coming instances: performance is maximized
			// if stream belongs to the src device. One for each expected transfer.
			stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
			gpuErrchk(cudaStreamCreate(&stream)):

			gpuErrchk(cudaEventCreate(&start));
			gpuErrchk(cudaEventCreate(&stop));
			gpuErrchk(cudaEventRecord(start, stream));

			for (rep = 0; rep < reps; ++rep){
				// inputs are: (void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, cudaStream_t stream = 0)
				gpuErrchk(cudaMemcpyPeerAsync(d_All[dst]->data, dst, d_All[src]->data, src,
				 size, stream));
			}

			gpuErrchk(cudaStreamSynchronize(stream));
		  gpuErrchk(cudaEventRecord(stop, stream));
			gpuErrchk(cudaEventSynchronize(stop));
			gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

			results[res_ctr++] = (reps * size) / (1.0e6 * milliseconds ) ;
			printf("%.3lf Gb/s \n", result);

			gpuErrchk(cudaEventDestroy(start));
			gpuErrchk(cudaEventDestroy(stop));
		}

		// Now a single memcpy between host and device:
		stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
		gpuErrchk(cudaStreamCreate(&stream)):

		gpuErrchk(cudaEventCreate(&start));
		gpuErrchk(cudaEventCreate(&stop));
		gpuErrchk(cudaEventRecord(start, stream));

		for (rep = 0; rep < reps; ++rep){
			// inputs are: (void *dst, const void *src, size_t count, enum cudaMemcpyKind	kind, cudaStream_t stream)
			gpuErrchk(cudaMemcpy(d_h->data, d_All[src]->data, size, cudaMemcpyDeviceToHost, stream));
		}

		gpuErrchk(cudaStreamSynchronize(stream));
		gpuErrchk(cudaEventRecord(stop, stream));
		gpuErrchk(cudaEventSynchronize(stop));
		gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

		results[res_ctr++] = (reps * size) / (1.0e6 * milliseconds ) ;
		printf("%.3lf Gb/s \n", result);

		gpuErrchk(cudaEventDestroy(start));
		gpuErrchk(cudaEventDestroy(stop));
	}

	// Print the results:
	int i, j, pr_ctr;
	for (i = 0; i < devicecount; ++i){
		for (j = 0; j < devicecount; ++j){
				if (j == i) printf(" X ");
				else {
						printf("%6.3lf", results[pr_ctr]);
				}
		}
		printf("\n")
	}

	printf("Cleaning up resources");
	kuhdaFreeM(h_A, 'k');
	for (i = 0; i < devicecount; ++i){
		kuhdaFreeM(d_All[device], 'c');
	}

	gpuErrchk(cudaDeviceReset());
	return 0;
}
