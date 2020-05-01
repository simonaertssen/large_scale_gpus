#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gpu_timer.h"

#define DEBUG 0
#define THREADS 32
#define DEVIDE_ID 0
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Debug mode only
int PRINT = 0;

// Global vars
GpuTimer timer;
double *d_A, *d_C, *h_A, *h_C;

// Utility functions declaration
void cleanUp();
void gpuAssert(cudaError_t, const char *, int);
void print_matrix(double const * const, const int, const int);

// Kernel declarations
__global__ void fill_matrix(double * const, const int, const int);
__global__ void matrix_equals_to(double const * const, const double, const int, const int);
__global__ void gpu_mul(double const * const, double * const, const int, const int);


/**
* Main function
**/ 
int main(int argc, char *argv[]) {
				
	int rows_A, cols_A, rows_C, cols_C;

	if (argc < 2) {
		fprintf(stderr, "Please provide rows, cols of matrix A\n");
		exit(1);
	}
	 
	rows_A = atoi(argv[1]);
	cols_A = atoi(argv[2]);
	rows_C = cols_C = cols_A;		// A**T*A = C
	
	// debug only
	if (argc > 3) {
		PRINT = atoi(argv[3]);
	}
	
	// Set cuda device
	gpuErrchk(cudaSetDevice(DEVIDE_ID));
	
	h_A = h_C = NULL;
	d_A = d_C = NULL;
	
	if (PRINT) {
		// Allocate A, C matrices on Host
		if ((h_A = (double *)malloc(rows_A * cols_A * sizeof(double))) == NULL ||
			(h_C = (double *)malloc(rows_C * cols_C * sizeof(double))) == NULL) {
			fprintf(stderr, "Host allocation error\n");
			cleanUp();
			exit(1);
		}
	}

	// Allocate A, C matrices on Device
	gpuErrchk(cudaMalloc(&d_A, rows_A * cols_A * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_C, rows_C * cols_C * sizeof(double)));
	
	dim3 block(THREADS, THREADS);
	dim3 grid(ceil(((float)cols_A)/block.x), ceil(((float)rows_A)/block.y));
	fill_matrix<<<grid, block>>>(d_A, rows_A, cols_A);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (PRINT) {
		printf("A =\n");
		gpuErrchk(cudaMemcpy(h_A, d_A, rows_A * cols_A * sizeof(double), cudaMemcpyDeviceToHost));
		print_matrix(h_A, rows_A, cols_A);
	}

	// Grid dimmensions for multiplication
	grid = dim3(ceil(((float)cols_C)/block.x), ceil(((float)rows_C)/block.y));
	
	// Perform the multiplications	
	timer.Start();
	gpu_mul<<<grid, block>>>(d_A, d_C, rows_A, cols_A);
	timer.Stop();
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
			
	if (PRINT) {
		printf("C =\n");
		gpuErrchk(cudaMemcpy(h_C, d_C, rows_C * cols_C * sizeof(double), cudaMemcpyDeviceToHost));
		print_matrix(h_C, rows_C, cols_C);
	}
	printf("Time elapsed: %f ms\n", timer.Elapsed());

	#if DEBUG
	double value = 0.;
	for (int i = 1; i < rows_A; ++i) {
		value += i * i;
	}	
	matrix_equals_to<<<grid, block>>>(d_C, value, rows_C, cols_C);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	#endif
	
	cleanUp();
	return 0;
}


/**
* Fill matrix sets the value of each element in matrix A to it's row number
* for debugging purposes
**/
__global__ void fill_matrix(double * const A, const int rows, const int cols) {
	
	const int row = blockIdx.y * blockDim.y + threadIdx.y,
			  col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows && col < cols) {
		A[row * cols + col] = row;
	}
}


/**
* Matrix equals to kernel checks if each element in the matrix is equal to
* the provided value. If not then it stops the block's execution.
**/
__global__ void matrix_equals_to(double const * const A, const double value, const int rows, const int cols) {
	
	const int row = blockIdx.y * blockDim.y + threadIdx.y,
			  col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && col < cols) {
		assert(A[row * cols + col] == value);
	}
}


/**
* Gpu mul kernel performs the calculation of C = A**T*A
**/
__global__ void gpu_mul(double const * const A, double * const C, const int rows_A, const int cols_A) {
	
	//Each Thread computes one element of C
	double C_element = 0.0;
	const int 	row = blockIdx.y * blockDim.y + threadIdx.y,
				col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < cols_A && col < cols_A) 
	{
		for (int k = 0; k < rows_A; ++k) {
			C_element += A[k * cols_A + row] * A[k * cols_A + col];
		}
		C[row * cols_A + col] = C_element;
	}
}


/**
* Utility function print_matrix, prints the contents of a matrix A
**/
void print_matrix(double const * const A, const int rows, const int cols) {
	int i, j;
	for (i = 0; i < rows; ++i) {
		for (j = 0; j < cols; ++j) {
			printf("%.2lf ", A[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}


/**
* Utility function cleanUp, deallocates dynamic memory on host and device
**/
void cleanUp() {
	// free GPU memory
	if (d_A) cudaFree(d_A);
	if (d_C) cudaFree(d_C);	

	// Free CPU memory
	if (h_A) free(h_A);
	if (h_C) free(h_C);
}


/**
* Utility function gpuAssert, checks success of cuda function calls
* and exits on failure
**/
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  cleanUp();
      exit(code);
   }
}