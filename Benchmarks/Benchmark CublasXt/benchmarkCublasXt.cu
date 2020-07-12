#include <stdio.h>
#include "../../DIEKUHDA/kuhda.h"
#include <cublasXt.h>
#include "cuda.h"

#define NUMTHREADSBUFF 16

// With this script, we aim to benchmark the performance of CublasXt. The maximum size 

// Run with nvcc -o benchmarkCublasXt -O3 -lcublas -Xcompiler -fopenmp ../DIEKUHDA/kuhda.cu benchmarkCublasXt.cu && ./benchmarkCublasXt

#define LOG(X,Y) fprintf(logfile, "%s, %s(%d) " #X " " #Y "\n", __TIMESTAMP__, __FILE__, __LINE__);

int main(int argc, char *argv[]) {
	// Regulate input:
	unsigned long n = 0, blockdim = 0;

	// Set matrix size
    if (argc > 1){
		n = (unsigned long)atoi(argv[1]);
		blockdim = n/2;
    }

    // Set tile size
    if (argc > 2){
        blockdim = (unsigned long)atoi(argv[2]);
	}
	if (blockdim > 4192) blockdim = 4192;

	// Find GPU info and adjust tile size
	int device_count;
	gpuErrchk(cudaGetDeviceCount(&device_count));
	// kuhdaAdjustTileSizeForAvailableMemory(device_count, n, blockdim);

	FILE *logfile = fopen("logfile_benchmarkCublasXt.txt", "a");
	// freopen("logfile_benchmarkCublasXt.txt","a",stdout);
	FILE *output = fopen("results_benchmarkCublasXt.txt", "a");
	if (logfile == NULL || output == NULL) {
		fclose(logfile);
		fclose(output);
    return 1;
  	}
	LOG(START, SUCCES);
	printf("n = %lu, blockdim = %lu\n", n, blockdim);

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

	int devices[device_count];
	for (int device = 0; device < device_count; ++device) devices[device] = device;
	CUBLASCHECK(cublasXtDeviceSelect(handle, device_count, devices));
	CUBLASCHECK(cublasXtSetBlockDim(handle, blockdim));
	CUBLASCHECK(cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED));


	double alpha = 1.0, beta  = 0.0;

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
