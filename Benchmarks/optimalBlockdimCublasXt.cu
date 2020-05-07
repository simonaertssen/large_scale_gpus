#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "cuda.h"

// With this script, we aim to measure the performance of cublasXt on the DTU cluster for different block dimensions.
// Run with nvcc -O3 -lcublas ../DIEKUHDA/kuhda.cu optimalBlockdimCublasXt.cu && ./a.out

// Easy cleanup of existing structures;
void cleanup(cublasXtHandle_t handle, matrix h_A, matrix h_B, matrix h_C, MatMultimer timer);
#define LOG(X, Y) fprintf(logfile, #X ": Time:%s, File:%s(%d) " #Y  "\n", __TIMESTAMP__, __FILE__, __LINE__)

int main(int argc, char *argv[]) {
	// Regulate input:
	int blockdim;
	unsigned long n;
	if (argc == 3){
		printf("Succesfully received input")
		blockdim = atoi(argv[1]);
		n = strtoul(argv[1]);
	} else {
		printf("Error receiving input")
		return -1;
	}

	FILE *logfile = fopen("logfile_optimalBlockdimCublasXt.txt", "a");
	FILE *output = fopen("results_optimalBlockdimCublasXt.txt", "a");
	if (logfile == NULL || output == NULL) {
		fclose(logfile);
		fclose(output);
    return 1;
  }

	LOG(START, n);

	// Find GPU info
	int device_count, status = 0;
	cudaErrchk(cudaGetDeviceCount(&device_count));

	// Allocate matrices
	unsigned long m = n, k = n;
	matrix *h_A  = kuhdaMallocMP1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocMP1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocMP(n, n); // empty C matrix

	// Allocate the timer:
	MatMultimer timer;

  // Perform the multiplications with CublasXt
  cublasXtHandle_t handle;
  status = cublasXtCreate(&handle);
  if (status != 0) {
		LOG(ERROR, "CUBLASXT handle creation failure");
		cleanup(handle, h_A, h_B, h_C, timer);
    return -1;
	}

  int devices[4] = {0, 1, 2, 3};
  if (cublasXtDeviceSelect(handle, device_count, devices) != 0) {
		LOG(ERROR, "CUBLASXT device selection error");
		cleanup(handle, h_A, h_B, h_C, timer);
    return -1;
  }

	if (cublasXtSetBlockDim(handle, blockdim) != 0) {
		LOG(ERROR, "CUBLASXT block dim selection error");
		cleanup(handle, h_A, h_B, h_C, timer);
    return -1;
  }

	double alpha = 1.0, beta  = 0.0;
  timer.Start();

  status = cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, h_A->data, m, h_B->data, k, &beta, h_C->data, m);
  if (status != 0) {
    LOG(ERROR, "CUBLASXT DGEMM error");
		cleanup(handle, h_A, h_B, h_C, timer);
    return -1;
  }

  // Wait for multiplications to finish
	timer.Stop()
	double timingResult = timer.GFLOPS_DGEMM(m, n, k);
	fprintf(output, "%d,%lf.5\n", n, timingResult);

	// Test the result for mistakes
	status = kuhdaTestMsilent(0, n, 0, n, h_C, 0);
	if (status != 0) {
    LOG(ERROR, "Unit test failed: result contains wrong components");
		cleanup(handle, h_A, h_B, h_C, timer);
    return -1;
  }

	// free all variables
	cleanup(handle, h_A, h_B, h_C, timer);
	LOG(SUCCESS, "Test finished succesfully");
	fclose(logfile);
	fclose(output);
	return 0;
}

void cleanup(cublasXtHandle_t handle, matrix h_A, matrix h_B, matrix h_C, MatMultimer timer){
	if (handle) cublasXtDestroy(handle);
	if (h_A) kuhdaFreeM(h_A, 'p');
	if (h_B) kuhdaFreeM(h_B, 'p');
	if (h_C) kuhdaFreeM(h_C, 'p');
	delete timer;
}
