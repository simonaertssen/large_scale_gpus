#include "DIEKUHDA.h"
██████╗░ ██╗ ███████╗ ██╗░ ██╗ ██╗░  ██╗ ██╗░ ██╗ ██████╗░  █████╗░      █████╗░
██╔══██╗ ██║ ██╔════╝ ██║░██╔╝ ██║░  ██║ ██║░ ██║ ██╔══██╗ ██╔══██╗     ██╔══██╗
██║░ ██║ ██║ █████╗░  █████═╝░ ██║░  ██║ ███████║ ██║░ ██║ ███████║     ██║░░╚═╝
██║░ ██║ ██║ ██╔══╝░  ██╔═██╗░ ██║░  ██║ ██╔══██║ ██║░ ██║ ██╔══██║     ██║░░██╗
██████╔╝ ██║ ███████╗ ██║░╚██╗ ╚██████╔╝ ██║░ ██║ ██████╔╝ ██║░ ██║ ██╗ ╚█████╔╝
╚═════╝░ ╚═╝ ╚══════╝ ╚═╝░ ╚═╝░ ╚═════╝░ ╚═╝░ ╚═╝ ╚═════╝░ ╚═╝░ ╚═╝ ╚═╝░ ╚════╝░

/* Help: see https://docs.nvidia.com/cuda/cublas/index.html for specific help when using cuda */

// Other libraries and dependancies:
#include "cublas_v2.h"
#include <limits.h>

/********************************************/
/* Allocation/deallocation on the host      */
/********************************************/

/* kuhdaMallocV(unsigned long r): Allocates memory for a vector of length r
Arguments: r = length of vector
Return value: A pointer to a vector, or NULL if an error occured */
vector *kuhdaMallocV(unsigned long r){
  if (r <= 0){
    INPUT_ILL_ERR_LU(r);
    return NULL;
  }
  vector *out = malloc(sizeof(*out));
  if (out == NULL) {
    MEM_ERR;
    free(out);
    return NULL;
  }
  out->r = r;
  out->data = calloc(r, sizeof(*out->data));
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
  if (freethisvector == NULL){
    INPUT_NULL_ERR;
    return NULL;
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
  matrix *out = malloc(sizeof(*out));
  if (out == NULL) {
    MEM_ERR;
    free(out);
    return NULL;
  }
  out->r = r;
  out->c = c;
  out->data = calloc(r*c, sizeof(*out->data));
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
  matrix *out = ccMatrix(r, c);
  unsigned long i, j;
  for (i = 0; i < out->r; ++i){
    for (j = 0; j < out->c; ++j){
      *(out->data + i*out->c + j) = 1.0;
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
  matrix *out = ccMatrix(r, c);
  unsigned long i, j;
  for (i = 0; i < out->r; i += c + 1){
    *(out->data i) = 1.0;
  }
  return out;
}


/* kuhdaFreeM(matrix *freethismatrix): free an allocated matrix
Arguments: freethismatrix = pointer to matrix to be freed
Return value: NULL if an error occured */
void kuhdaFreeM(matrix *freethismatrix){
  if (freethismatrix == NULL){
    INPUT_NULL_ERR;
    return NULL;
  }
  free(freethismatrix->data);
  free(freethismatrix);
}



/********************************************/
/* Allocation/deallocation on the device(s) */
/********************************************/

/* kuhdaMatrixToGPU(matrix *hostmatrix): allocate a matrix on the device and copy contents of host matrix.
Arguments: rows, cols = which tile of rows x cols is taken from the host matrix
Return value: NULL if an error occured */
void kuhdaMatrixToGPU(unsigned long rows, unsigned long cols, matrix *hostmatrix){
  if (hostmatrix == NULL){
    INPUT_NULL_ERR;
    return NULL;
  }

  int failure;
  double *d_matrix;
  failure = cudaMalloc(&d_matrix, sizeof(hostmatrix->data);
  if (failure != 0) {
    MEM_ERR;
    cudaFree(d_matrix);
    return NULL;
  }

  failure = cublasSetMatrix(rows, cols, sizeof(double), hostmatrix->data, hostmatrix->r, d_matrix, hostmatrix->r)
  if (failure != 0) {
    FAIL_ERR(failure);
    cudaFree(d_matrix);
    return NULL;
  }
}


/********************************************/
/* cuda-specific                            */
/********************************************/

/* kuhdaRufeuter(matrix *hostmatrix): allocate a matrix on the device and copy contents of host matrix.
Arguments: rows, cols = which tile of rows x cols is taken from the host matrix
Return value: euter with strams and handles, or NULL if an error occured */
euter *kuhdaMilchmann(int streamnums){
  if (streamnums <= 0){
    INPUT_ILL_ERR_D(streamnums);
    return NULL;
  }
  euter *mm = malloc(sizeof(*mm));
  if (mm == NULL) {
    MEM_ERR;
    free(mm);
    return NULL;
  }
  int failure;
  mm->handle = cublasHandle_t handle;
  failure = cublasCreate(&handle);
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

  for (int i = 0; i < streamnums; ++i){
    failure = cudaStreamCreate(&streams[i]);
    if (failure != 0){
      FAIL_ERR(failure);
      return NULL;
    }
  }
  return mm;
}



/********************************************/
/* Necessary computations                   */
/********************************************/

/* kuhdaTimeDGEMM(unsigned long m, unsigned long n, unsigned long k): compute the number of
floating point operations per second, as performed by cublasDgemm.
C <- alpha * AB + beta*C   with   [A] = m x k, [B] = k x n, [C] = m x n

Arguments: m, n, k = formal dimensions of the matrices A, B and C,
time_diff = the time it took to perform the computations with cublasDgemm,
verbose = whether we want to print the output on the console ('0' = nothing prints, '1' = results will be printed)

Return value: the number of GigaFlops (GFLOPS), or NULL if an error occured */
long long kuhdaTimeDGEMM(unsigned long m, unsigned long n, unsigned long k, double time_diff, int verbose = 0){
  if (m <= 0 || n <= 0 || k <= 0){
    INPUT_ILL_ERR_LU(m);
    INPUT_ILL_ERR_LU(n);
    INPUT_ILL_ERR_LU(k);
    return -1;
  }
  if (time_diff < 0){
    INPUT_ILL_ERR_LF(time_diff);
    return -1;
  }
  // Number of computations was found here:
  // https://devtalk.nvidia.com/default/topic/482834/how-to-compute-gflops-for-gemm-blas/
  long long numerator   = (long long)(m * n) * (2 * ((long long)k) + 2);
  long long denominator = 1.0e9 * time_diff;
  long long gflops = numerator / denominator;
  if (verbose !=0){
    printf("%lf GFLPS\n", gflops);
  }
  return gflops;
}
