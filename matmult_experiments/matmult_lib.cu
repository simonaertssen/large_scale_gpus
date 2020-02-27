#include <stdio.h>
#include <stdlib.h>

extern "C" {
	
    #include <cblas.h>

	void matmult_lib(int m,int n,int k,double *A,double *B,double *C)
	//void matmult_lib(int m,int n,int k,double *A_f,double *B_f,double *C_f)
	{
        /*
        double *A, *B, *C;
        A = (double*)malloc(sizeof(double)* m * k);
        B = (double*)malloc(sizeof(double)* k * n);
        C = (double*)malloc(sizeof(double)* m * n);
        */
		//cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,k,B,n,0.0,C,n); // Works.
		
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,m,k,1.0,B,n,A,k,0.0,C,n); // Works also, maybe faster!?

		//cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,m,B,k,0.0,C,m); // Works.
        /*
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,K,
                    alpha,A,M,
                    B,K,
                    beta,C,M);
        */
		//cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,m,k,1.0,&B[0][0],n,&A[0][0],k,0.0,&C[0][0],n); // Works also, maybe faster!?
        /*
        free(A);
        free(B);
        free(C);
        */
	}

    /*
	void matmult_nat(int m,int n,int k,double *A,double *B,double *C)
	{
		for (int i = 0; i < n; ++i) 
			for (int j = 0; j < m; ++j)
				C[i+n*j] = 0;

		for (int i = 0; i < n; ++i) 
			for (int j = 0; j < m; ++j)
				for (int kk = 0; kk < k; ++kk)
					C[i+n*j] += A[kk+j*k] * B[i+kk*n];
	}
    */
}
