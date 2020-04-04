#include <stdio.h>
#include "kuhda.h"


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

matrix *kuhdaMallocMdiagP(unsigned long r, unsigned long c){
	matrix *out = kuhdaMallocMP(r, c);
	unsigned long i;
	for (i = 0; i < r*c; i += c + 1){
		*(out->data + i) = 1.0;
	}
	return out;
}

matrix *kuhdaMallocDeviceM(unsigned long r, unsigned long c){
	if (r <= 0){
        INPUT_ILL_ERR_LU(r);
        return NULL;
    }
    if (c <= 0){
        INPUT_ILL_ERR_LU(c);
        return NULL;
    }
    
    matrix *out = NULL;
    gpuErrchk(cudaMalloc((void**)&out, sizeof(*out)));
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


int main(){
    printf("Test");
    unsigned long n = 10;
    matrix *test = kuhdaMallocMdiagP(n,n);
    kuhdaPrintM(test);
    return 0;
}