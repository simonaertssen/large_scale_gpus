#include <stdio.h>
#include "../../DIEKUHDA/kuhda.h"
#include <cublasXt.h>
#include "cuda.h"

// With this script, we aim to measure the performance of cublasXt on the DTU cluster for different block dimensions.
// Run with nvcc -o optimalBlockdimCublasXt -O3 -lcublas ../DIEKUHDA/kuhda.cu optimalBlockdimCublasXt.cu && ./optimalBlockdimCublasXt

#define LOG(X,Y) fprintf(logfile, "%s, %s(%d) " #X " " #Y "\n", __TIMESTAMP__, __FILE__, __LINE__);

int main(int argc, char *argv[]) {
	// Regulate input:
	unsigned long n, blockdim;

	if (argc == 2){
		blockdim = (unsigned long)atoi(argv[1]);
	} else if (argc == 3) {
		n = (unsigned long)atoi(argv[1]);
		blockdim = (unsigned long)atoi(argv[2]);
	} 
	adjustedblockdim = blockdim;

	// Find GPU info, and only adjust block dimension if there is not enough memory
	int device_count;
	gpuErrchk(cudaGetDeviceCount(&device_count));
	// kuhdaAdjustTileSizeForAvailableMemory(device_count, n, adjustedblockdim);
	// if (adjustedblockdim < blockdim) blockdim = adjustedblockdim;
	// if (blockdim > 8192) blockdim = 8192;

	FILE *logfile = fopen("logfile_optimalBlockdimCublasXt.txt", "a");
	// freopen("logfile_optimalBlockdimCublasXt.txt","a",stdout);
	FILE *output = fopen("results_optimalBlockdimCublasXt.txt", "a");
	if (logfile == NULL || output == NULL) {
		fclose(output);
    return 1;
  	}
	LOG(START, SUCCES);
	printf("n = %zu, blockdim = %zu\n", n, blockdim);

	// Allocate matrices
	unsigned long m = n, k = n;
	matrix *h_A = kuhdaMallocMdiag(n, n); // matrix A as a diagonal matrix
    matrix *h_B = kuhdaMallocMdiag(n, n); // matrix B to be filled with specific values for specific testing
    matrix *h_C = kuhdaMallocM(n, n);     // matrix C will contain results: same values at each spot as in b
    unsigned long i, j;
    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF)
	for (i = 0; i < h_B->r; ++i){
		for (j = 0; j < h_B->c; ++j){
            h_B->data[i*h_B->c + j] = (i + j) * 0.1 + i;
        }
    }

	// Allocate the timer:
	MatMulTimer timer;

  	// Perform the multiplications with CublasXt
	cublasXtHandle_t handle;
	CUBLASCHECK(cublasXtCreate(&handle));
	CUBLASCHECK(cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED));

	int devices[device_count];
	for (int device = 0; device < device_count; ++device) devices[device] = device;
	CUBLASCHECK(cublasXtDeviceSelect(handle, device_count, devices));
	CUBLASCHECK(cublasXtSetBlockDim(handle, blockdim));

	double alpha = 1.0, beta  = 0.0;

	printf("Computation start.\n");

  	timer.Start();

  	CUBLASCHECK(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, h_A->data, m, h_B->data, k, &beta, h_C->data, m));

  	// Wait for multiplications to finish
	timer.Stop();
    double timingResult = timer.GFLOPS_DGEMM(m, n, k);
    printf("GFLOPS = %.0lf..\n", timingResult);

    // Test the result for mistakes
    printf("Checking results. ");
    double abserror = 0.0, totalerror = 0.0;
    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF) reduction(+:totalerror)
	for (i = 0; i < h_B->r; ++i){
		for (j = 0; j < h_B->c; ++j){
            abserror = fabs(h_B->data[i*h_B->c + j] - h_C->data[i*h_C->c + j]);
            totalerror += abserror;
            if (abserror > 10e-6) {
                // printf("Failure: B[%d] = %1.4e != C[%d] = %1.4e\n", i*h_B->c + j, h_B->data[i*h_B->c + j], i*h_C->c + j, h_C->data[i*h_C->c + j]);
                break;
            }
        }
    }
    if (totalerror < 10e-6) printf("Succes.\n");	
	fprintf(output, "%zu, %d, %.1lf\n", n, blockdim, timingResult);

	// free all variables
	cublasXtDestroy(handle);
	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
	kuhdaFreeM(h_C, 'k');

	LOG(STOP, SUCCES);
	fclose(logfile);
	fclose(output);
	return 0;
}
