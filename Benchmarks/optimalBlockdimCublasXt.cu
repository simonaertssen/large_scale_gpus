#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "cuda.h"

// With this script, we aim to measure the performance of cublasXt on the DTU cluster for different block dimensions.
// Run with nvcc -O3 -lcublas ../DIEKUHDA/kuhda.cu optimalBlockdimCublasXt.cu && ./a.out

int main(int ) {
	// Find GPU info
	int device_count, status = 0;
	gpuErrchk(cudaGetDeviceCount(&device_count));


	return 0;
}
