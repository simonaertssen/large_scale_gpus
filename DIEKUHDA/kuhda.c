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
/* Allocation/deallocation on the host			*/
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
	type == 'c' ? gpuErrchk(cudaFree(freethismatrix->data)) : free(freethismatrix->data);
	if (freethismatrix != NULL) free(freethismatrix);
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


/********************************************/
/* Allocation/deallocation on the device(s) */
/********************************************/

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


/********************************************/
/* cuda-specific							*/
/********************************************/

/* kuhdaMilkCan(int streamnums): 
Arguments: number of streams
Return value: euter with strams and handles, or NULL if an error occured */
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

/*gpuAssert(cudaError_t code, const char *file, int line): check for cuda errors.
Arguments: code = cudafunction to be wrapped around, file and line = place where the error occured */
cudaError_t gpuAssert(cudaError_t code, const char *file, int line){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s in file %s at line %d\n", cudaGetErrorString(code), file, line);
      //exit(code);
   }
   return code;
}



/********************************************/
/* Necessary computations					*/
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
