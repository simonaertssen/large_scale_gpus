#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gpu_timer.h"

#define DEBUG 0
#define THREADS 32
#define TILE_WIDTH 48
#define BLOCK_WIDTH 16
#define REG_WIDTH (TILE_WIDTH/BLOCK_WIDTH)

#define DEVIDE_ID 0
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define min(a, b) ((a)<(b)?(a):(b))

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
__global__ void gpu_mul(double const * const __restrict__, double * const __restrict__, const int, const int);


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

	// Blocking factor for multiplication
	block = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
	grid = dim3(ceil(((float)cols_C)/TILE_WIDTH), ceil(((float)rows_C)/TILE_WIDTH));
	
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
	
	block = dim3(THREADS, THREADS);
	grid = dim3(ceil(((float)cols_C)/block.x), ceil(((float)rows_C)/block.y));
	
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
__global__ void gpu_mul(double const * const __restrict__  A, double * const __restrict__ C, const int rows_A, const int cols_A)
{
	// A_T * A multiplication results to a symmetric matrix C,
	// so we only calculate half of the matrix ( upper triangular in our case )
	// and copy the results to their symmetric indices in C.
	if (blockIdx.x < blockIdx.y) return;

	const int idx = threadIdx.x;
	const int idy = threadIdx.y;

	const int blx = blockIdx.x;
	const int bly = blockIdx.y;

	// +1 padding for reducing bank conflicts 
	__shared__ double sA_T[BLOCK_WIDTH][TILE_WIDTH + 1];
	__shared__ double sA[TILE_WIDTH][BLOCK_WIDTH + 1];

	// Registers rA_T, rA used in inner product calculation stored in rC
	double rC[REG_WIDTH][REG_WIDTH];
	double rA_T[REG_WIDTH];
	double rA[REG_WIDTH];

	// Registers rA_T, rA copying the next tile from device to shared memory 
	double ra_T[REG_WIDTH];
	double ra[REG_WIDTH];
	
	// Tile offsets in A, A_T for every thread as well as bound calculation
	double const * offs_dA_T = A + blx * TILE_WIDTH + idy * cols_A + idx;
	ptrdiff_t boundA_T = (rows_A * cols_A) - (blx * TILE_WIDTH + idy * cols_A + idx) - 1;

	double const * offs_dA = A + bly * TILE_WIDTH + idy * cols_A + idx;
	ptrdiff_t boundA = (rows_A * cols_A) - (bly * TILE_WIDTH + idy * cols_A + idx) - 1;
	
	int m, n, k, temp;

	// Initialization of register rC[][] = 0 
	#pragma unroll
	for (n = 0; n < REG_WIDTH; ++n) {
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			rC[n][m] = 0.0;
		}
	}

	// Load from GLOBAL -> SHARED memory the first Tile of A_T
	#pragma unroll
	for (m = 0; m < TILE_WIDTH; m += BLOCK_WIDTH) {
		sA_T[idy][m + idx] = offs_dA_T[min(m, boundA_T)];
	}

	// Load from GLOBAL -> SHARED memory the first Tile of A
	#pragma unroll
	for (m = 0; m < TILE_WIDTH; m += BLOCK_WIDTH) {
		sA[m + idx][idy] = offs_dA[min(m, boundA)];
	}

	__syncthreads();
	
	// Loop for each Tile
	for (temp = 0; temp < rows_A - BLOCK_WIDTH; temp += BLOCK_WIDTH)
	{
		// Calculate new offsets & bounds
		offs_dA_T += BLOCK_WIDTH * cols_A;
		boundA_T  -= BLOCK_WIDTH * cols_A;

		offs_dA   += BLOCK_WIDTH * cols_A;
		boundA 	  -= BLOCK_WIDTH * cols_A;

		// Load from GLOBAL -> REGISTERS memory the next Tile of A_T
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			ra_T[m] = offs_dA_T[min(m * BLOCK_WIDTH, boundA_T)];
		}
		
		// Load from GLOBAL -> REGISTERS memory the next Tile of A
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			ra[m] = offs_dA[min(m * BLOCK_WIDTH, boundA)];
		}

		// Multiplication A_T*A
		#pragma unroll
		for (k = 0; k < BLOCK_WIDTH; ++k)
		{
			// Load from SHARED -> REGISTERS memory part of sA_T
			#pragma unroll
			for (m = 0; m < REG_WIDTH; ++m) {
				rA_T[m] = sA_T[k][m * BLOCK_WIDTH + idx];
			}
			
			// Load from SHARED -> REGISTERS memory part of sA
			#pragma unroll
			for (n = 0; n < REG_WIDTH; ++n) {
				rA[n] = sA[n * BLOCK_WIDTH + idy][k];
			}

			// Compute an store the result into rC registers
			#pragma unroll
			for (n = 0; n < REG_WIDTH; ++n) {
				#pragma unroll
				for (m = 0; m < REG_WIDTH; ++m) {
					rC[n][m] += rA_T[m] * rA[n];
				}
			}
		}

		__syncthreads();
		
		// Load from REGISTERS-> SHARED the next tile of A_T
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			sA_T[idy][m * BLOCK_WIDTH + idx] = ra_T[m];
		}
		
		// Load from REGISTERS-> SHARED the next tile of A
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			sA[m * BLOCK_WIDTH + idx][idy] = ra[m];
		}

		__syncthreads();
	}
	
	// Calculate total remaining elements
	temp = rows_A - temp;
	
	#pragma unroll
	for (k = 0; k < temp; ++k)
	{
		// Load from SHARED -> REGISTERS memory part of sA_T
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			rA_T[m] = sA_T[k][m * BLOCK_WIDTH + idx];
		}
		
		// Load from SHARED -> REGISTERS memory part of sA
		#pragma unroll
		for (n = 0; n < REG_WIDTH; ++n) {
			rA[n] = sA[n * BLOCK_WIDTH + idy][k];
		}

		// Compute an store the final result into rC registers
		#pragma unroll
		for (n = 0; n < REG_WIDTH; ++n) {
			#pragma unroll
			for (m = 0; m < REG_WIDTH; ++m) {
				rC[n][m] += rA_T[m] * rA[n];
			}
		}
	}

	int coord_dCn, coord_dCm;

	// Store from REGISTERS -> GLOBAL memory results in rC 
	#pragma unroll
	for (n = 0; n < REG_WIDTH; ++n) {
		coord_dCn = bly * TILE_WIDTH + n * BLOCK_WIDTH + idy;
		#pragma unroll
		for (m = 0; m < REG_WIDTH; ++m) {
			coord_dCm = blx * TILE_WIDTH + m * BLOCK_WIDTH + idx;
			if (coord_dCm < cols_A && coord_dCn < cols_A) {
				C[coord_dCn * cols_A + coord_dCm] = rC[n][m];
				C[coord_dCm * cols_A + coord_dCn] = rC[n][m]; 	// Symmetric results
			}
		}
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
void inline gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		cleanUp();
		exit(code);
	}
}