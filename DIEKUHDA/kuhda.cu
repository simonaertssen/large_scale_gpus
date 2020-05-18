#include "kuhda.h"
/*
$$$$$$$\  $$$$$$\ $$$$$$$$\ $$\   $$\ $$\   $$\ $$\   $$\ $$$$$$$\   $$$$$$\      $$$$$$\
$$  __$$\ \_$$  _|$$  _____|$$ | $$  |$$ |  $$ |$$ |  $$ |$$  __$$\ $$  __$$\    $$  __$$\
$$ |  $$ |  $$ |  $$ |      $$ |$$  / $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ /  $$ |   $$ /  \__|
$$ |  $$ |  $$ |  $$$$$\    $$$$$  /  $$ |  $$ |$$$$$$$$ |$$ |  $$ |$$$$$$$$ |   $$ |
$$ |  $$ |  $$ |  $$  __|   $$  $$<   $$ |  $$ |$$  __$$ |$$ |  $$ |$$  __$$ |   $$ |
$$ |  $$ |  $$ |  $$ |      $$ |\$$\  $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |   $$ |  $$\
$$$$$$$  |$$$$$$\ $$$$$$$$\ $$ | \$$\ \$$$$$$  |$$ |  $$ |$$$$$$$  |$$ |  $$ |$$\\$$$$$$  |
\_______/ \______|\________|\__|  \__| \______/ \__|  \__|\_______/ \__|  \__|\__|\______/

Help: see https://docs.nvidia.com/cuda/cublas/index.html for specific help when using cuda */

// Other libraries and dependancies:
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include "cuda.h"
#include <omp.h>

/********************************************/
/* Allocation/deallocation on the HOST			*/
/********************************************/

/* kuhdaMallocV(unsigned long r): Allocates memory for a vector of length r
Arguments: r = length of vector
Return value: A pointer to a vector, or NULL if an error occured */
vector *kuhdaMallocV(unsigned long r){
	if (r <= 0){
		INPUT_ILL_ERR_LU(r);
		return NULL;
	}
	vector *out = (vector *) malloc(sizeof(*out));
	if (out == NULL) {
		MEM_ERR;
		free(out);
		return NULL;
	}
	out->r = r;
	out->data = (double *) calloc(r, sizeof(*out->data));
	if (out->data == NULL) {
		MEM_ERR;
		free(out->data);
		free(out);
		return NULL;
	}
	return out;
}


/* kuhdaFreeV(vector *freethisvector): free an allocated vector
Arguments: freethisvector = pointer to vector to be freed
Return value: NULL if an error occured */
void kuhdaFreeV(vector *freethisvector){
	printf("freeing this vector\n");
	if (freethisvector == NULL){
		INPUT_NULL_ERR;
	}
	free(freethisvector->data);
	free(freethisvector);
}


/* kuhdaMallocM(unsigned long r, unsigned long c):
Allocates memory for a matrix of length r*c. The matrix will be filled with zeros.
Remember that DIEKUHDA matrices (type matrix) are 1D arrays!
Arguments: r = number of matrix rows, c = number of matrix columns
Return value: A pointer to a matrix, or NULL if an error occured */
matrix *kuhdaMallocM(unsigned long r, unsigned long c){
	if (r <= 0 || c <=0 ){
		INPUT_ILL_ERR_LU(r);
		INPUT_ILL_ERR_LU(c);
		return NULL;
	}
	matrix *out = (matrix *) malloc(sizeof(*out));
	if (out == NULL) {
		MEM_ERR;
		free(out);
		return NULL;
	}
	out->r = r;
	out->c = c;
	out->data = (double *) calloc(r*c, sizeof(double));
	if (out->data == NULL) {
		MEM_ERR;
		free(out->data);
		free(out);
		return NULL;
	}
	return out;
}

/* kuhdaMallocM1(unsigned long r, unsigned long c):
Allocates memory for a matrix of length r*c. The matrix will be filled with ones.
Remember that DIEKUHDA matrices (type ccMatrix) are 1D arrays!
Arguments: r = number of matrix rows, c = number of matrix columns
Return value: A pointer to a matrix, or NULL if an error occured */
matrix *kuhdaMallocM1(unsigned long r, unsigned long c){
	matrix *out = kuhdaMallocM(r, c);
	unsigned long i, j;
	for (i = 0; i < r; ++i){
		for (j = 0; j < c; ++j){
			*(out->data + i*c + j) = 1.0;
		}
	}
	return out;
}

/* kuhdaMallocMdiag(unsigned long r, unsigned long c):
Allocates memory for a matrix of length r*c. The matrix will be a diagonal matrix.
Remember that DIEKUHDA matrices (type ccMatrix) are 1D arrays!
Arguments: r = number of matrix rows, c = number of matrix columns
Return value: A pointer to a matrix, or NULL if an error occured */
matrix *kuhdaMallocMdiag(unsigned long r, unsigned long c){
	matrix *out = kuhdaMallocM(r, c);
	unsigned long i;
	for (i = 0; i < r*c; i += c + 1){
		*(out->data + i) = 1.0;
	}
	return out;
}


/* kuhdaFreeM(matrix *freethismatrix): free an allocated matrix
Arguments: freethismatrix = pointer to matrix to be freed, type = cuda or kuhda
Return value: NULL if an error occured */
void kuhdaFreeM(matrix *freethismatrix, char type){
	if (freethismatrix == NULL) INPUT_NULL_ERR;

	switch(type) {

   	case 'c': // a kuhda matrix with data member on a device
      	gpuErrchk(cudaFree(freethismatrix->data));
		free(freethismatrix);
      	break;

	case 'p': // a kuhda matrix with data member pinned on the host
    	gpuErrchk(cudaFreeHost(freethismatrix->data));
		gpuErrchk(cudaFreeHost(freethismatrix));
      	break;

   	case 'k': // a kuhda matrix with data member on the host
   		free(freethismatrix->data);
   		free(freethismatrix);
		break;
	}
}


/********************************************/
/*    Vector / Matrix printing utilities    */
/********************************************/
void kuhdaPrintV(vector *printthisvector){
	if (printthisvector == NULL){
		INPUT_NULL_ERR;
	}
	unsigned long i;
	printf("[");
	for (i = 0; i < printthisvector->r; ++i){
		printf("%5.3lf", printthisvector[i]);
	}
	printf("]\n");
}


void kuhdaPrintM(matrix *printthismatrix){
	if (printthismatrix == NULL){
		INPUT_NULL_ERR;
	}
	unsigned long i,j;
	for (i = 0; i < printthismatrix->r; ++i){
		printf("|");
		for (j = 0; j < printthismatrix->c; ++j){
			printf("%6.2lf", printthismatrix->data[i*printthismatrix->c + j]);
		}
		printf("|\n");
	}
}

// Test whether all elements of this matrix are equal to its' dimensions.
// Only for the result of multiplication on square ones!
void kuhdaTestM(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *testhismatrix){
	if (testhismatrix == NULL){
		INPUT_NULL_ERR;
	}
	if (rowstart - rowstop != colstart - colstop){
		printf("The testfunction is only deigned for square tiles.");
		return;
	}

	unsigned long i,j,value = 0,as_we_would_expect = (int)(rowstop - rowstart);
	for (i=rowstart; i<rowstop; ++i){
		for (j=colstart; j<colstop; ++j){
			value = (int)testhismatrix->data[i*testhismatrix->c + j];
			if (value != as_we_would_expect){
				printf("The matrix does not contain the expected results at ");
				printf("(%d, %d) = %d != %d\n", i,j, value, as_we_would_expect);
				FAIL_ERR(value);
				return;
			}
		}
	}
	printf("Test succeeded. No errors.\n");
}

// Test whether all elements of this matrix are equal to its' dimensions.
// Only for the result of multiplication on square ones!
int kuhdaTestMsilent(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *testhismatrix, int verbose){
	if (testhismatrix == NULL){
		INPUT_NULL_ERR;
		return -1;
	}
	if (rowstart - rowstop != colstart - colstop){
		if (verbose != 0) printf("The testfunction is only deigned for square tiles.");
		return -1;
	}

	unsigned long i, j, value = 0, as_we_would_expect = (int)(rowstop - rowstart);
	for (i = rowstart; i < rowstop; ++i){
		for (j = colstart; j < colstop; ++j){
			value = (int)testhismatrix->data[i*testhismatrix->c + j];
			if (value != as_we_would_expect){
				if (verbose != 0) printf("The matrix does not contain the expected results at (%d, %d) = %d != %d\n", i,j, value, as_we_would_expect);
				FAIL_ERR(value);
				return -1;
			}
		}
	}
	if (verbose != 0) printf("Test succeeded. No errors.\n");
	return 0;
}



/********************************************/
/* Allocation/deallocation on the DEVICE(S) */
/********************************************/

/* kuhdaMallocDeviceM: cudaMalloc of a [r * c] matrix structure on the device */
matrix *kuhdaMallocDeviceM(unsigned long r, unsigned long c){
	if (r <= 0){
        INPUT_ILL_ERR_LU(r);
        return NULL;
    }
    if (c <= 0){
        INPUT_ILL_ERR_LU(c);
        return NULL;
    }

    matrix *out = (matrix *) malloc(sizeof(*out));
    if (out == NULL) {
			MEM_ERR;
			gpuErrchk(cudaFree(out));
			return NULL;
		}

	out->r = r;
	out->c = c;
    out->data = NULL;
	gpuErrchk(cudaMalloc((void**)&out->data, r*c*sizeof(double)));
    if (out->data == NULL) {
		MEM_ERR;
		gpuErrchk(cudaFree(out->data));
	    gpuErrchk(cudaFree(out));
		return NULL;
	}
	return out;
}


/* PINNED allocation routine for matrix of dimension [r * c] */
matrix *kuhdaMallocMP(unsigned long r, unsigned long c){
	if (r <= 0){
        INPUT_ILL_ERR_LU(r);
        return NULL;
    }
    if (c <= 0){
        INPUT_ILL_ERR_LU(c);
        return NULL;
    }

    matrix *out = NULL;
    gpuErrchk(cudaMallocHost((void**)&out, sizeof(*out)));
    if (out == NULL) {
		MEM_ERR;
		gpuErrchk(cudaFreeHost(out));
		return NULL;
	}

	out->r = r;
	out->c = c;
    out->data = NULL;
	gpuErrchk(cudaHostAlloc((void**)&out->data, r*c*sizeof(double), cudaHostAllocPortable));
    if (out->data == NULL) {
		MEM_ERR;
		gpuErrchk(cudaFreeHost(out->data));
	    gpuErrchk(cudaFreeHost(out));
		return NULL;
	}
	return out;
}

/* PINNED allocation for [r * c] matrix of onesss */
matrix *kuhdaMallocMP1(unsigned long r, unsigned long c){
	matrix *out = kuhdaMallocMP(r, c);
	unsigned long i, j;
	for (i = 0; i < r; ++i){
		for (j = 0; j < c; ++j){
			*(out->data + i*c + j) = 1.0;
		}
	}
	return out;
}

/* PINNED allocation for [r * c] identity matrix */
matrix *kuhdaMallocMdiagP(unsigned long r, unsigned long c){
	matrix *out = kuhdaMallocMP(r, c);
	unsigned long i;
	for (i = 0; i < r*c; i += c + 1){
		*(out->data + i) = 1.0;
	}
	return out;
}


// Fill and test with value
void kuhdaFillWithValue(matrix *A, double value){
	unsigned long i, j;
	for (i = 0; i < A->r; ++i){
		for (j = 0; j < A->c; ++j){
			A->data[i*A->c + j] = value;
		}
	}
}

void kuhdaTestForValue(matrix *A, double value, int verbose){
	unsigned long i, j;
	int result = 0;
	for (i = 0; i < A->r; ++i){
		for (j = 0; j < A->c; ++j){
			if (A->data[i*A->c + j] != value){
				fprintf(stderr,"%s: encountered wrong value %.2lf at (%zu,%zu)\n",__func__, A->data[i*A->c + j], i, j);
				result = -1;
				return;
			}
		}
	}
	if (result == 0 && verbose == 1) printf("%s tested correctly for value %.2lf\n", __func__, value);
}

void kuhdaFillDiagonalWithValue(matrix *A, double value){
	unsigned long i;
	for (i = 0; i < A->r*A->c; i += A->c + 1) A->data[i] = value;
}

void kuhdaTestDiagonalForValue(matrix *A, double value, int verbose){
	unsigned long i;
	int result = 0;
	for (i = 0; i < A->r*A->c; i += A->c + 1){
		if (A->data[i] != value){
			fprintf(stderr,"%s: encountered wrong value %.2lf at (%zu,%zu)\n",__func__, i, i);
			result = -1;
			return;
		}
	}
	if (result == 0 && verbose == 1) printf("%s tested correctly for value %.2lf\n", __func__, value);
}

/********************************************/
/* 				 Data transfers 			*/
/********************************************/

/*
TileHostToGPU: memcopy tile of host matrix to device.
Arguments: dimensions / location of tile to be copied, pointers to hostmatrix & device-tile, streams
Return value: none
*/
void *TileHostToGPU(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *h_matrix, matrix *d_tile, cudaStream_t stream)
{
	if (h_matrix == NULL || d_tile == NULL) 	INPUT_NULL_ERR;
	if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
	if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
	if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
	if (stream == NULL) INPUT_NULL_ERR;

	unsigned long cols = colstop - colstart, i, j;
	cudaError_t failure;

	// allocate space (size of a single tile row) on the host:
	double *memacc = (double*)malloc(cols*sizeof(double));
	if (memacc == NULL){
		MEM_ERR;
		free(memacc);
		return 0;
	}

	// 'strided' copy, row by row
	for (i=rowstart; i<rowstop; ++i){
		for (j=colstart; j<colstop; ++j){
			// fill memacc with host-matrix data one (tile-)row at a time:
			memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
		}

		// Asynchronous copy to device
		// takes (d_arr, h_arr, nbytes, cudaMemcpyHostToDevice, stream)
		failure = gpuErrchk(cudaMemcpyAsync((void*) (&d_tile->data[0] + (cols * (i-rowstart))), memacc, cols*sizeof(double), cudaMemcpyHostToDevice, stream));

		if (failure != 0) {
			FAIL_ERR(failure);
			cudaFree(d_tile);
			}
	}
	return 0;
}

/*
TileGPUToHost: memcopy tile of device matrix to host.
Arguments: dimensions / location of tile to be copied, pointers to hostmatrix & device-tile, streams
Return value: none
*/
void *TileGPUToHost(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *d_tile, matrix *h_matrix, cudaStream_t stream)
{
	if (h_matrix == NULL || d_tile == NULL) 	INPUT_NULL_ERR;
	if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
	if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
	if (stream == NULL) INPUT_NULL_ERR;


	unsigned long cols = colstop - colstart, i, j;
	cudaError_t failure;

	double *memacc = (double*)malloc(cols*sizeof(double));
	if (memacc == NULL){
		MEM_ERR;
		free(memacc);
		return 0;
	}

	// 'strided' copy, row by row
	for (i=rowstart; i<rowstop; ++i){
		// takes (d_arr, h_arr, nbytes, cudaMemcpyHostToDevice, stream)
		failure = gpuErrchk(cudaMemcpyAsync(memacc, (void*) (&d_tile->data[0] + (cols * (i-rowstart))), cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
		for (j=colstart; j<colstop; ++j){
			h_matrix->data[i * h_matrix->c + j] = memacc[j-colstart];
			//memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
		}

		if (failure != 0) {
			FAIL_ERR(failure);
			cudaFree(d_tile);
		}
	}
	// printf("Tile copied successfully.\n");
	return 0;
}

/*
TileGPUAddToHost: memcopy tile of device matrix to host.
Arguments: dimensions / location of tile to be copied, pointers to hostmatrix & device-tile, streams
Return value: none
*/
void *TileGPUAddToHost(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *d_tile, matrix *h_matrix, cudaStream_t stream)
{
	if (h_matrix == NULL || d_tile == NULL) 	INPUT_NULL_ERR;
	if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
	if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
	if (stream == NULL) INPUT_NULL_ERR;


	unsigned long cols = colstop - colstart, i, j;
	cudaError_t failure;

	double *memacc = (double*)malloc(cols*sizeof(double));
	if (memacc == NULL){
		MEM_ERR;
		free(memacc);
		return 0;
	}

	// 'strided' copy, row by row
	for (i=rowstart; i<rowstop; ++i){
		// takes (d_arr, h_arr, nbytes, cudaMemcpyHostToDevice, stream)
		failure = gpuErrchk(cudaMemcpyAsync(memacc, (void*) (&d_tile->data[0] + (cols * (i-rowstart))), cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
		for (j=colstart; j<colstop; ++j){
			h_matrix->data[i * h_matrix->c + j] += memacc[j-colstart];
			//memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
		}

		if (failure != 0) {
			FAIL_ERR(failure);
			cudaFree(d_tile);
		}
	}
	// printf("Tile copied successfully.\n");
	return 0;
}



void kuhdaMatrixToHost(unsigned long rows, unsigned long cols, matrix *d_matrix, matrix *h_matrix){
	if (h_matrix == NULL || d_matrix == NULL){
			INPUT_NULL_ERR;
	}
	//int failure = cublasGetMatrix(rows, cols, sizeof(double), d_matrix->data, d_matrix->r, h_matrix->data, h_matrix->r);
	//int failure = cudaMemcpy2D(h_matrix->data, d_matrix->data, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
	cudaError_t failure = gpuErrchk(cudaMemcpy(h_matrix->data, d_matrix->data, rows*cols*sizeof(double), cudaMemcpyDeviceToHost));
	if (failure != 0){
		FAIL_ERR(failure);
	}
}


void kuhdaTileToHost(unsigned long rows, unsigned long cols, double *d_tile, matrix *h_matrix){
	if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
	if (rows != h_matrix->r) INPUT_ILL_ERR_LU(rows);
	if (cols != h_matrix->c) INPUT_ILL_ERR_LU(cols);

	//int failure = cublasGetMatrix(rows, cols, sizeof(double), d_matrix->data, d_matrix->r, h_matrix->data, h_matrix->r);
	//int failure = cudaMemcpy2D(h_matrix->data, d_matrix->data, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
	cudaError_t failure = gpuErrchk(cudaMemcpy(h_matrix->data, d_tile, rows*cols*sizeof(double), cudaMemcpyDeviceToHost));
	if (failure != 0){
		FAIL_ERR(failure);
	}
}


/*gpuAssert(cudaError_t code, const char *file, int line): check for cuda errors.
Arguments: code = cudafunction to be wrapped around, file and line = place where the error occured */
cudaError_t gpuAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: error in file %s, line %d\n", file, line);
  		fprintf(stderr,"code %d with reason %s\n", code, cudaGetErrorString(code));
    	exit(1);
   }
   return code;
}

// New definitions for error checks:
cublasStatus_t cublasAssert(cublasStatus_t error, const char *file, int line){
    if (error != CUBLAS_STATUS_SUCCESS){
      fprintf(stderr, "CUBLASCHECK: error in file %s, line %d \n", file, line);
      fprintf(stderr,"error code = %d\n", error);
      exit(1);
	}
	return error;
}


void kuhdaWarmup(int devicecount){
	// Sync current device
	cudaDeviceSynchronize();
	int device;
	for(device = 0; device < devicecount; ++device){
		GPUCHECK(cudaSetDevice(device));
		int *testint = 0;
		GPUCHECK(cudaMalloc((void**)&testint,sizeof(int)));
		GPUCHECK(cudaFree(testint));
		cudaDeviceSynchronize();
	}
}



/********************************************/
/* 			Necessary computations			*/
/********************************************/

/* kuhdaTimeDGEMM(unsigned long m, unsigned long n, unsigned long k): compute the number of
floating point operations per second, as performed by cublasDgemm.
C <- alpha * AB + beta*C	 with	 [A] = m x k, [B] = k x n, [C] = m x n

Arguments: m, n, k = formal dimensions of the matrices A, B and C,
time_diff = the time it took to perform the computations with cublasDgemm,
verbose = whether we want to print the output on the console ('0' = nothing prints, '1' = results will be printed)

Return value: the number of GigaFlops (GFLOPS), or NULL if an error occured */
long long kuhdaTimeDGEMM(matrix *d_matrix, int reps, int verbose){
	if (d_matrix == NULL){
		INPUT_NULL_ERR;
		return -1;
	}
	// Data for the computations:
	unsigned int m = d_matrix->r, k = d_matrix->r, n = d_matrix->c;
	double alpha = 1.0, beta  = 0.0;
	cublasHandle_t handle;
	int failure = cublasCreate(&handle);
	if (failure != 0){
		FAIL_ERR(failure);
		return -1;
	}
	cudaStream_t stream = (cudaStream_t) malloc(sizeof(cudaStream_t));
  	gpuErrchk(cudaStreamCreate(&stream));
	failure = cublasSetStream(handle, stream);
	if (failure != 0){
		FAIL_ERR(failure);
		return -1;
	}

	// Events for the dgemm timing:
	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));

	int rep = 0;
	gpuErrchk(cudaEventRecord(start, 0));
	gpuErrchk(cudaStreamSynchronize(0));
	for (rep = 0; rep < reps; ++rep){
		failure = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
			d_matrix->data, m, d_matrix->data, k, &beta, d_matrix->data, m);
		if (failure != 0){
			FAIL_ERR(failure);
			return -1;
		}
	}
	gpuErrchk(cudaStreamSynchronize(0));
  //gpuErrchk(cudaDeviceSynchronize()); // Not necessary when using cudaEvents
  gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));

	float milliseconds = 0;
	gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

	// Number of computations was found here:
	// https://devtalk.nvidia.com/default/topic/482834/how-to-compute-gflops-for-gemm-blas/
	long int numerator    = (long int)(m * n) * (2 * ((long long)k) + 2) * reps;
	long long denominator = 1.0e6 * milliseconds;
	long long gflops = numerator / denominator;
	if (verbose !=0){
		printf("%lu GFLPS\n", gflops);
	}
	// Clean up:
	cublasDestroy(handle);
	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(stop));
	return gflops;
}



/* kuhdamm(matrix *d_A_tile, matrix *d_B_tile, matrix *d_C_tile, int verbose):
perform matrix-matrix multiplication of tiles on a device, performed by cublasDgemm.
C <- alpha * AB + beta*C	 with	 [A] = m x k, [B] = k x n, [C] = m x n

Arguments: m, n, k = formal dimensions of the matrices A, B and C,
verbose = whether we want to print the output on the console
('0' = nothing prints, '1' = results will be printed)

Return value: the number of GigaFlops (GFLOPS), or NULL if an error occured */
int kuhdamm(matrix *d_A_tile, matrix *d_B_tile, matrix *d_C_tile, cudaStream_t stream, int verbose){
	if (d_A_tile == NULL || d_B_tile == NULL || d_C_tile == NULL){
		INPUT_NULL_ERR;
		return -1;
	}
	if (d_A_tile->r != d_C_tile->r || d_A_tile->c != d_B_tile->r || d_B_tile->c != d_C_tile->c){
		INPUT_ILL_ERR_D(d_A_tile->r);
		return DIEKUHDA_DIMENSION_MISMATCH;
	}
	if (stream == NULL) INPUT_NULL_ERR;

	// Data for the computations:
	unsigned int m = d_A_tile->r, k = d_A_tile->c, n = d_C_tile->c;
	double alpha = 1.0, beta  = 0.0;
	cublasHandle_t handle;
	int failure = cublasCreate(&handle);
	if (failure != 0){
		FAIL_ERR(failure);
		return -1;
	}

	failure = cublasSetStream(handle, stream);
	if (failure != 0){
		FAIL_ERR(failure);
		return -1;
	}
	failure = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
			d_A_tile->data, m, d_B_tile->data, k, &beta, d_C_tile->data, m);
	if (failure != 0){
		FAIL_ERR(failure);
		return -1;
	}
	cublasDestroy(handle);
	return 0;
}




//////////////////////////////////////////////////////////////////
// DEPRECATED - use allocation and data transfers separately... //
//////////////////////////////////////////////////////////////////

/* kuhdaMatrixToGPU(matrix *h_matrix): allocate a matrix on the device and copy contents of host matrix.
Arguments: rows, cols = which tile of rows x cols is taken from the host matrix
Return value: NULL if an error occured */
matrix *kuhdaMatrixToGPU(unsigned long rows, unsigned long cols, matrix *h_matrix){
	if (h_matrix == NULL){
		INPUT_NULL_ERR;
	}

	cudaError_t failure;
	matrix *d_matrix = kuhdaMallocM(rows, cols);
	// failure = gpuErrchk(cudaMalloc(&d_matrix->data, rows*cols*sizeof(double)));
	failure = gpuErrchk(cudaMalloc((void**)&d_matrix->data, rows*cols*sizeof(double))); // Tip from HH
	if (failure != 0) {
		MEM_ERR;
		kuhdaFreeM(d_matrix, 'k');
	} // rows, cols = which tile of rows x cols is taken from the host matrix

	//failure = cublasSetMatrix(rows, cols, sizeof(double*), h_matrix->data, rows, d_matrix->data, rows);
	//failure = cudaMemcpy2D(&h_matrix->data, &d_matrix->data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
	failure = gpuErrchk(cudaMemcpy(d_matrix->data, h_matrix->data, rows*cols*sizeof(double), cudaMemcpyHostToDevice));
	if (failure != 0) {
		FAIL_ERR(failure);
		cudaFree(d_matrix);
	}

	return d_matrix;
}

/* kuhdaTileToGPU(matrix *h_matrix): allocate a matrix on the device and copy contents of host matrix.
Arguments: rows, cols = which tile of rows x cols is taken from the host matrix
Return value: NULL if an error occured */
double *kuhdaTileToGPU(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, matrix *h_matrix){
	if (h_matrix == NULL) 	INPUT_NULL_ERR;
	if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
	if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);

	unsigned long rows = rowstop - rowstart, cols = colstop - colstart;
	unsigned long i, j;

	double *memacc = (double*)malloc(cols*sizeof(double));
	if (memacc == NULL){
		MEM_ERR;
	}
	cudaError_t failure;
	// matrix *d_tile = kuhdaMallocM(rows, cols);
	double *d_tile = NULL;

	// failure = gpuErrchk(cudaMalloc(&d_matrix->data, rows*cols*sizeof(double)));
	failure = gpuErrchk(cudaMalloc((void**)&d_tile, rows*cols*sizeof(double))); // Tip from HH
	if (failure != 0) {
		MEM_ERR;
		cudaFree(d_tile);
	//	kuhdaFreeM(d_matrix, 'k');
	} // rows, cols = which tile of rows x cols is taken from the host matrix

	//double* tilep = &d_tile[0];
	// 'strided' copy
	for (i=rowstart; i<rowstop; ++i){
		for (j=colstart; j<colstop; ++j){
				memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
		}
		//printf("%zu\n", cols * (i-rowstart) );
		// printf("%zu\n", sizeof( *(tilep + (cols * (i-rowstart))) ) );
		//failure = gpuErrchk(cudaMemcpy((void*)d_tile + (cols * (i-rowstart)), memacc, cols*sizeof(double), cudaMemcpyHostToDevice));
		failure = gpuErrchk(cudaMemcpy((void*) (&d_tile[0] + (cols * (i-rowstart))), memacc, cols*sizeof(double), cudaMemcpyHostToDevice));

		if (failure != 0) {
			FAIL_ERR(failure);
			cudaFree(d_tile);
		}
	}

	return d_tile;
}


/********************************************/
/* 				cuda-specific		  		*/
/********************************************/
/* kuhdaMilkCan(int streamnums):
Arguments: number of streams
Return value: euter with strams and handles, or NULL if an error occured */
/*
can *kuhdaMilkCan(int streamnums){
	if (streamnums <= 0){
		INPUT_ILL_ERR_D(streamnums);
		return NULL;
	}
	can *mm = (can *) malloc(sizeof(*mm));
	if (mm == NULL) {
		MEM_ERR;
		free(mm);
		return NULL;
	}
	int failure;
	failure = cublasCreate(&(mm->handle));
	if (failure != 0){
		FAIL_ERR(failure);
		return NULL;
	}
	mm->streams = (cudaStream_t *) malloc(streamnums*sizeof(cudaStream_t));
	if (mm->streams == NULL) {
		MEM_ERR;
		free(mm->streams);
		free(mm);
		return NULL;
	}
	int i;
	for (i = 0; i < streamnums; ++i){
		failure = cudaStreamCreate(&(mm->streams)[i]);
		if (failure != 0){
			FAIL_ERR(failure);
			return NULL;
		}
	}
	return mm;
}
*/
