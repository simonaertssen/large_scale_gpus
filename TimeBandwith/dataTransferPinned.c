#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas ../DIEKUHDA/kuhda.c dataTransferTiming2.c && ./a.out

int main(){
	// Set verbose to 0 to mute output
	int verbose = 0;

	unsigned long n = 8000, size = n * n * sizeof(double);
	double printmem = size / 1.0e9;
	printf("Masuring: n = %d\nSending %.4f GB of memory between host and devices\n", n, printmem);

	// Get the overall structure setup
	int devicecount;
	struct cudaDeviceProp prop;
	gpuErrchk(cudaGetDeviceCount(&devicecount));

	// Containers for host and device matrices
	// matrix *h_A = kuhdaMallocMdiag(n, n), *d_All[devicecount];

	// Now with pinned memory:
	// matrix *h_A = (matrix *) malloc(sizeof(*h_A)), *d_All[devicecount];
	matrix *h_A = NULL, *d_All[devicecount];
	gpuErrchk(cudaMallocHost((void**)&h_A, sizeof(*h_A)));
	h_A->r = n;
	h_A->c = n;
	h_A->data = NULL;
	//gpuErrchk(cudaMallocHost((void**)&h_A->data, n*n*sizeof(double)));
	//gpuErrchk(cudaHostAlloc((void**)&h_A->data, n*n*sizeof(double), cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&h_A->data, n*n*sizeof(double), cudaHostAllocPortable));

	int i,j;
	// See if we do get and set the right devices:
	if (verbose == 1) printf("Allocating memory ..");
	for (i = 0; i < devicecount; ++i){
		gpuErrchk(cudaSetDevice(i));

		d_All[i] = kuhdaMatrixToGPU(n, n, h_A);

		// Make device able to send and receive data from peers:
		for (j = 0; j < devicecount; ++j){
			if (j == i) continue;
			gpuErrchk(cudaDeviceEnablePeerAccess(j, 0));
		}
	}

	// Time connections between devices:
	float milliseconds, results[16], result;
	int res_ctr = 0;
	cudaEvent_t start, stop;
	cudaStream_t stream;

	// Now the device - device loop
	int src, dst, reps = 1, rep, accessible;
	for (src = 0; src < devicecount; ++src){
		// Set the current device:
		gpuErrchk(cudaSetDevice(src));
		gpuErrchk(cudaGetDeviceProperties(&prop, src));
		if (verbose == 1) printf("Timing connections of device %s: %d of %d\n", prop.name, src, devicecount);

		for (dst = 0; dst < devicecount; ++dst){
			// Check if not on the diagonal:
			if (src == dst) continue;
			// Check if access is set:
			gpuErrchk(cudaDeviceCanAccessPeer(&accessible, dst, src));
			if (accessible == 0){
				if (verbose == 1) printf("Device %d cannot access device %d\n", dst, src);
				continue;
			}
			if (verbose == 1) printf("Transfer %d to %d: ", src, dst);

			// Now continue:
			// Create the stream for the coming instances: performance is maximized
			// if stream belongs to the src device. One for each expected transfer.
			// stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
			gpuErrchk(cudaStreamCreate(&stream));

			gpuErrchk(cudaEventCreate(&start));
			gpuErrchk(cudaEventCreate(&stop));
			gpuErrchk(cudaEventRecord(start, stream));

			for (rep = 0; rep < reps; ++rep){
				// inputs are: (void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, cudaStream_t stream = 0)
				gpuErrchk(cudaMemcpyPeerAsync(d_All[dst]->data, dst, d_All[src]->data, src, size, stream));
			}

			gpuErrchk(cudaStreamSynchronize(stream));
		  	gpuErrchk(cudaEventRecord(stop, stream));
			gpuErrchk(cudaEventSynchronize(stop));
			gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

			result = (reps * size) / (1.0e6 * milliseconds ) ;
			if (verbose == 1) printf("%.3lf Gb/s \n", result);
			results[res_ctr++] = result;

			gpuErrchk(cudaEventDestroy(start));
			gpuErrchk(cudaEventDestroy(stop));
		}

		// Now a single memcpy between host and device:
		if (verbose == 1) printf("Transfer %d to h: ", src);
		// stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
		//gpuErrchk(cudaStreamCreate(stream));

		gpuErrchk(cudaEventCreate(&start));
		gpuErrchk(cudaEventCreate(&stop));
		gpuErrchk(cudaEventRecord(start, stream));

		for (rep = 0; rep < reps; ++rep){
			// inputs: (void *dst, const void *src, size_t count, enum cudaMemcpyKind	kind, cudaStream_t stream)
			gpuErrchk(cudaMemcpyAsync(h_A->data, d_All[src]->data, size, cudaMemcpyDeviceToHost, stream));
		}

		gpuErrchk(cudaStreamSynchronize(stream));
		gpuErrchk(cudaEventRecord(stop, stream));
		gpuErrchk(cudaEventSynchronize(stop));
		gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

		result = (reps * size) / (1.0e6 * milliseconds ) ;
		if (verbose == 1) printf("%.3lf Gb/s \n", result);
		results[res_ctr++] = result;

		gpuErrchk(cudaEventDestroy(start));
		gpuErrchk(cudaEventDestroy(stop));
		gpuErrchk(cudaStreamDestroy(stream));

		// Endline:
		if (verbose == 1) printf("\n");
	}

	// Print the results:
	if (verbose == 1) printf("Gathering results ..\n");
	printf("	GPU0	GPU1	GPU2	GPU3	host    \n");

	int pr_ctr = 0;
	for (i = 0; i < devicecount; ++i){
		i == devicecount ? printf("host ") : printf("GPU%d ", i);
		for (j = 0; j < devicecount + 1; ++j){
			if (j == i) {
				printf("     X  ");
			} else {
				printf("%8.2lf", results[pr_ctr++]);
			}
		}
		printf("\n");
	}
	printf("host  ");
	for (i = 0; i < devicecount; ++i){
		printf(" %6.2lf ", results[4*(i+1) - 1]);
	}
	printf("    X  \n");

	if (verbose == 1) printf("Cleaning up resources ..\n");
	gpuErrchk(cudaFreeHost(h_A->data));
	gpuErrchk(cudaFreeHost(h_A));

	for (i = 0; i < devicecount; ++i){
		kuhdaFreeM(d_All[i], 'c');
	}
	gpuErrchk(cudaDeviceReset());
	return 0;
}
