/*
$$$$$$$\  $$$$$$\ $$$$$$$$\ $$\   $$\ $$\   $$\ $$\   $$\ $$$$$$$\   $$$$$$\      $$\   $$\
$$  __$$\ \_$$  _|$$  _____|$$ | $$  |$$ |  $$ |$$ |  $$ |$$  __$$\ $$  __$$\     $$ |  $$ |
$$ |  $$ |  $$ |  $$ |      $$ |$$  / $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ /  $$ |    $$ |  $$ |
$$ |  $$ |  $$ |  $$$$$\    $$$$$  /  $$ |  $$ |$$$$$$$$ |$$ |  $$ |$$$$$$$$ |    $$$$$$$$ |
$$ |  $$ |  $$ |  $$  __|   $$  $$<   $$ |  $$ |$$  __$$ |$$ |  $$ |$$  __$$ |    $$  __$$ |
$$ |  $$ |  $$ |  $$ |      $$ |\$$\  $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |    $$ |  $$ |
$$$$$$$  |$$$$$$\ $$$$$$$$\ $$ | \$$\ \$$$$$$  |$$ |  $$ |$$$$$$$  |$$ |  $$ |$$\ $$ |  $$ |
\_______/ \______|\________|\__|  \__| \______/ \__|  \__|\_______/ \__|  \__|\__|\__|  \__|

                           /|                       /|
                          | \           __ _ _     / ;
                    ___    \ \   _.-"-" `~"\  `"--' /
                _.-'   ""-._\ ""   ._,"  ; "\"--._./
            _.-'       \./    "-""", )  ~"  |
           / ,- .'          ,     '  `o.  ;  )
           \ ;/       '                 ;   /
            |/        '      |      \   '   |
            /        |             J."\  ,  |
           "         :       \   .'  : | ,. _)
           |         |     /     f |  |`--"--'
            \_        \    \    / _/  |
             \ "-._  _.|   (   j/; -'/
              \  | "/  (   |   /,    |
               | \  |  /\  |\_///   /
               \ /   \ | \  \  /   /
                ||    \ \|  |  |  |
                ||     \ \  |  | /
                |\      |_|/   ||
                L \       ||   ||
                `"'       |\   |\
                          ( \. \ `.
                          |_ _\|_ _\
                            "    "

DTU Special course: Large Scale GPU Computing
Authors: Simon Aertssen (s181603) and Louis Hein (s181573)
Supervisors: Bernd Dammann and Hans Henrik Brandenborg Soerensen

DIEKUHDA (pronounce "d-cuda"):  Basic data structures, allocation/deallocation
routines, and input/output routines for matrices, to be used in the project

Version: 1.0 13/03/2020
*/

#ifndef DIEKUHDA_DEFINE
#define DIEKUHDA_DEFINE

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"


/* Macro definitions */
#define DIEKUHDA_FAILURE -1
#define DIEKUHDA_SUCCESS 0
#define DIEKUHDA_MEMORY_ERROR 1
#define DIEKUHDA_FILE_ERROR 2
#define DIEKUHDA_ILLEGAL_INPUT 3
#define DIEKUHDA_DIMENSION_MISMATCH 4
#define MEM_ERR fprintf(stderr,"%s: failed to allocate memory\n",__func__)
#define FAIL_ERR(x) fprintf(stderr,"%s: failure detected, error %d\n",__func__, x)
#define gpuErrchk(ans)  gpuAssert((ans), __FILE__, __LINE__)
#define GPUCHECK(ans)  gpuAssert((ans), __FILE__, __LINE__)
#define CUBLASCHECK(ans)  cublasAssert((ans), __FILE__, __LINE__)
#define INPUT_NULL_ERR fprintf(stderr,"%s: received NULL pointer as input\n",__func__)
#define INPUT_ILL_ERR_D(x) fprintf(stderr,"%s: received illegal input %d\n",__func__, x)
#define INPUT_ILL_ERR_LF(x) fprintf(stderr,"%s: received illegal input %lf\n",__func__, x)
#define INPUT_ILL_ERR_LU(x) fprintf(stderr,"%s: received illegal input %u\n",__func__, x)

/* _____ _______ _____  _    _  _____ _______ _    _ _____  ______  _____
  / ____|__   __|  __ \| |  | |/ ____|__   __| |  | |  __ \|  ____|/ ____|
 | (___    | |  | |__) | |  | | |       | |  | |  | | |__) | |__  | (___
  \___ \   | |  |  _  /| |  | | |       | |  | |  | |  _  /|  __|  \___ \
  ____) |  | |  | | \ \| |__| | |____   | |  | |__| | | \ \| |____ ____) |
 |_____/   |_|  |_|  \_\\____/ \_____|  |_|   \____/|_|  \_\______|_____/
*/
/* Structure representing a DIEKUHDA vector */
typedef struct vector {
  unsigned long r;   /* number of elements */
  double * data;     /* pointer to array of length r */
} vector;

/* Structure representing a DIEKUHDA matrix */
typedef struct matrix {
  unsigned long r;   /* number of rows */
  unsigned long c;   /* number of columns */
  double * data;     /* pointer to array of length r*c */
} matrix;

/* deprecated structure: unused
typedef struct can {
  cublasHandle_t handle;   // pointer type to an opaque structure holding the cuBLAS library context
  cudaStream_t *streams;   // stream IDs
} can;
*/

/*______ _    _ _   _  _____ _______ _____ ____  _   _  _____
|  ____| |  | | \ | |/ ____|__   __|_   _/ __ \| \ | |/ ____|
| |__  | |  | |  \| | |       | |    | || |  | |  \| | (___
|  __| | |  | | . ` | |       | |    | || |  | | . ` |\___ \
| |    | |__| | |\  | |____   | |   _| || |__| | |\  |____) |
|_|     \____/|_| \_|\_____|  |_|  |_____\____/|_| \_|_____/
*/
/* Allocation/deallocation on the host*/
// vectors
vector *kuhdaMallocV(unsigned long r);
void kuhdaFreeV(vector *freethisvector);
// matrices
matrix *kuhdaMallocM(unsigned long r, unsigned long c);
matrix *kuhdaMallocM1(unsigned long r, unsigned long c);
matrix *kuhdaMallocMdiag(unsigned long r, unsigned long c);
void kuhdaFreeM(matrix *freethismatrix, char type);

// Pinned allocation routines:
matrix *kuhdaMallocMP(unsigned long r, unsigned long c);
matrix *kuhdaMallocMP1(unsigned long r, unsigned long c);
matrix *kuhdaMallocMdiagP(unsigned long r, unsigned long c);
matrix *kuhdaMallocDeviceM(unsigned long r, unsigned long c);

// Filling and checking:
void kuhdaFillWithValue(matrix *A, double value);
void kuhdaTestForValue(matrix *A, double value, int verbose);
void kuhdaFillDiagonalWithValue(matrix *A, double value);
void kuhdaTestDiagonalForValue(matrix *A, double value, int verbose);

/* Printing */
void kuhdaPrintV(vector *freethisvector);
void kuhdaPrintM(matrix *printhismatrix);
void kuhdaPrintDeviceM(matrix *printthismatrix);
void kuhdaTestM(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *testhismatrix);
int kuhdaTestMsilent(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *testhismatrix, int verbose);


/* Allocation/deallocation on the device(s)*/
matrix *kuhdaMatrixToGPU(unsigned long rows, unsigned long cols, matrix *h_matrix);
double *kuhdaTileToGPU(unsigned long rowstart,unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *h_matrix);
void kuhdaMatrixToHost(unsigned long rows, unsigned long cols, matrix *d_matrix, matrix *h_matrix);
void kuhdaTileToHost(unsigned long rows, unsigned long cols, double *d_tile, matrix *h_matrix);
void TileHostToGPU(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *h_matrix, matrix *d_tile, cudaStream_t stream);
void TileGPUToHost(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *d_tile, matrix *h_matrix, cudaStream_t stream);
void TileGPUAddToHost(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *d_tile, matrix *h_matrix, cudaStream_t stream);


/* CUDA-specific */
void kuhdaWarmup(int devicecount);
void kuhdaWarmupDevice(int device); // for omp parallel calls
size_t kuhdaAvailableMemoryOnCurrentDevice();
unsigned int kuhdaAdjustTileSizeForAvailableMemory(int devicecount, unsigned int matrixsize, unsigned int tilesize);
cudaError_t gpuAssert(cudaError_t code, const char *file, int line);
cublasStatus_t cublasAssert(cublasStatus_t error, const char *file, int line);


/* Necessary computations*/
// A timer to record the necessary computations when performing DGEMM
struct MatMulTimer
{
	MatMulTimer() {
    cudaStreamCreate(&stream);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~MatMulTimer() {
	}

	void Start() {
		cudaEventRecord(start, stream);
	}

	void Stop() {
		cudaEventRecord(stop, stream);
	}

	void Release(){
		cudaStreamDestroy(stream);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	double GFLOPS_DGEMM(unsigned int m, unsigned int n, unsigned int k) {
		// Calculate the number of operations necessary for a matrix multiplication A * B with [A] = m x k and [B] = k x n
	  	// See https://forums.developer.nvidia.com/t/how-to-compute-gflops-for-gemm-blas/20218/6
		float elapsedtime;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime, start, stop);
	  	long unsigned int numerator = (long unsigned int)(m * n) * (long unsigned int)(2 * k + 2); 		// [GFLOP]
    	double denominator = (double) 1.0e6 * elapsedtime;												// [1/s]
    	// printf("elapsed time = %lf\n", elapsedtime);
		return (double) numerator / denominator;
  	}

  double GFLOPS_MM(int m, int n, int k) {
	  // Calculate the number of operations necessary for a matrix multiplication A * B with [A] = m x k and [B] = k x n
	  // See https://software.intel.com/en-us/articles/a-simple-example-to-measure-the-performance-of-an-intel-mkl-function
		float elapsedtime;
	  	long int M = (long int)m, N = (long int)n, K = (long int)k;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime, start, stop);
	  	long unsigned int numerator = (M * N) * (K - 2);
    	double denominator = (double) 1.0e6 * elapsedtime;
		return (double) numerator / denominator;
	}

	private :
		cudaEvent_t start;
		cudaEvent_t stop;
    	cudaStream_t stream; // aka mainstream
};


struct Timer
{
	Timer() {
    	cudaStreamCreate(&stream);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~Timer() {
	}

	void Release(){
		cudaStreamDestroy(stream);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start, stream);
	}

	float Stop() {
		cudaEventRecord(stop, stream);
    	cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime, start, stop);
    	return elapsedtime;
	}

	private :
		cudaEvent_t start;
		cudaEvent_t stop;
    	cudaStream_t stream;
		float elapsedtime;
};

int kuhdamm(matrix *d_A_tile, matrix *d_B_tile, matrix *d_C_tile, cudaStream_t stream, cublasHandle_t handle);
int kuhdammson(matrix *d_A_tile, matrix *d_B_tile, matrix *d_C_tile, cudaStream_t stream, cublasHandle_t handle);
long long kuhdaTimeDGEMM(matrix *d_matrix, int reps, int verbose);

#endif
