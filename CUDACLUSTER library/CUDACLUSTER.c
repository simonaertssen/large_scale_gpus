#include "CUDACLUSTER.h"


/********************************************/
/* Allocation/deallocation on the host      */
/********************************************/


/* ccVector(unsigned long r): Allocates memory for a vector of length r
Arguments:
  r = length of vector
Return value:
  A pointer to a vector, or NULL if an error occured
*/
vector *ccVector(unsigned long r){
    if (r <= 0){
        INPUT_ILL_ERR(r);
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
		return NULL;
	}
	return out;
}


/* ccFreeVector(vector *freethisvector): free an allocated vector
Arguments:
  freethisvector = pointer to vector to be freed
Return value:
  NULL if an error occured
*/
void ccFreeVector(vector *freethisvector){
    if (freethisvector == NULL) return NULL;
    free(freethisvector->data);
    free(freethisvector);
}


/* ccMatrix(unsigned long r, unsigned long c): 
Allocates memory for a matrix of length r*c.
The matrix will be filled with zeros.
Remember that CUDACLUSTER matrices (type ccMatrix) are 1D arrays!
Arguments:
  r = number of matrix rows
  c = number of matrix columns
Return value:
  A pointer to a matrix, or NULL if an error occured
*/
matrix *ccMatrix(unsigned long r, unsigned long c){
    if (r <= 0 || c <=0 ){
        INPUT_ILL_ERR(r);
        INPUT_ILL_ERR(c);
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
		return NULL;
	}
	return out;
}


/* ccOnesMatrix(unsigned long r, unsigned long c): 
Allocates memory for a matrix of length r*c.
The matrix will be filled with ones.
Remember that CUDACLUSTER matrices (type ccMatrix) are 1D arrays!
Arguments:
  r = number of matrix rows
  c = number of matrix columns
Return value:
  A pointer to a matrix, or NULL if an error occured
*/
matrix *ccOnesMatrix(unsigned long r, unsigned long c){
    matrix *out = ccMatrix(r, c);
    unsigned long i, j;
    for (i = 0; i < out->r; ++i){
        for (j = 0; j < out->c; ++j){
            *(out->data + out->c + j) = 1.0;
        }
    }
    return out;
}


/* ccFreeMatrix(matrix *freethismatrix): free an allocated matrix
Arguments:
  freethismatrix = pointer to matrix to be freed
Return value:
  NULL if an error occured
*/
void ccFreeMatrix(matrix *freethismatrix){
    if (freethismatrix == NULL) return NULL;
    free(freethismatrix->data);
    free(freethismatrix);
}



/********************************************/
/* Allocation/deallocation on the device(s) */
/********************************************/

matrix *ccMatrixToGPU(matrix *hostmatrix){
    if (hostmatrix == NULL){
        
    }
    int result;
    double *todevice;
    if (cudaMalloc(&todevice, sizeof(hostmatrix->data)) != 0){
        fprintf(stderr, "CudaMalloc failed: matrix is of size %ldGB which is larger than 16GB (V100 memory).\n", n_squared * sizeof(double) / 10e9);
        exit(-1);
        }
}
