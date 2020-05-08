#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublasXt.h>

#define MAX_GPUS 4

float getGflops (int, float);

int main(int argc, char** argv){
    double* A;
    double* B;
    double* C;
    
    double alpha = 1.0;
    double beta = 0.0;
    int i, j;        
    float time;
    
    bool correct = true;
    
    int num_devices = MAX_GPUS;
    int device_IDs[MAX_GPUS];
    for ( i = 0 ; i < MAX_GPUS; i++)
      device_IDs[i] = i; 
    
    cudaEvent_t start, stop; 
  
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    cublasStatus_t status;
    cublasXtHandle_t handle;
    
    // set matrix size 
    int width = 32768;
    if (argc > 1){
        width = atoi(argv[1]);
	if ( width < 0 || width > 40960 ) {
	    width = 10240;
	}
    }
    
    // set tile size 
    int block_size = 4096;
    if (argc > 2){
        block_size = atoi(argv[2]);
	if ( block_size < 0 || block_size > width ) {
	    block_size = 1024;
	}
    }
	  
	  
    // set number of GPUs to be used
    if (argc > 3){
        num_devices = atoi(argv[3]);
	if ( ( num_devices > MAX_GPUS ) || ( num_devices < 1 ) ) {
	    printf ("\nNumber of GPUs to use must be in {1..4}"); 
	    return 0;
	}
		
    }

    // banner
  printf ("\n\nGPU DGEMM Exercise\n");
  printf (    "==========================================\n");
  printf (  "\nTiled Matrix-Matrix Multiplication\n");
  printf (    "Using NVIDIA cublasXt Library\n");


    A = (double*) malloc (width * width * sizeof(double));
    B = (double*) malloc (width * width * sizeof(double));
    C = (double*) malloc (width * width * sizeof(double));
    

    /* Init A and B. */
    for (i = 0; i < width; ++i) {
      for (j = 0; j < width; ++j) {
	A[i * width + j] = i + j;
	B[i * width + j] = (i == j) ? 2.0 : 0.0;
      }
    }

    /* Now prepare the call to CUBLAS */
    status = cublasXtCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }
    
    // start timing
    cudaEventRecord(start, 0);

        
   /* Perform calculation
      -  host pointers can be passed !
      - for MultiGPU usage the device IDs have to be passed
    */
    
    status =  cublasXtDeviceSelect( handle, num_devices, device_IDs );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! cublasXtDeviceSelect error\n");
        return EXIT_FAILURE;
    }
    status = cublasXtSetBlockDim( handle, block_size );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! cublasXtSetBlockDim error\n");
        return EXIT_FAILURE;
    }
    status = cublasXtSetCpuRatio( handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, 0.f );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! cublasXtSetCpuRatio error\n");
        return EXIT_FAILURE;
    }
    status = cublasXtSetPinningMemMode( handle, CUBLASXT_PINNING_ENABLED );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! cublasXtSetPinningMemMode error\n");
        return EXIT_FAILURE;
    }
   
    status = cublasXtDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_T, width, width, width, &alpha, A,
        width, B, width, &beta, C, width );
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    

    // Stop timing and get elapsed time for kernel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); 
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    
    // check results
    for ( i = 0 ; i < width; ++i) {
      for ( j = 0 ; j < width; ++j ) {
	if ( C[i * width + j] != 2 * A[i * width + j] ) {
	    correct = false;
	    break;
	}
	if (correct == false)
	  break;
      }
    }
    
    if ( correct ) {
      printf("\nCall to cublasXtDgemm took %lf ms (width: %d)\n", time, width);
      printf("\nUsing %d GPU(s), block_size: %d", num_devices, block_size);
      printf("\nDevice IDs: ");
      for (i = 0 ; i < num_devices; ++i)
	printf ("%d ", device_IDs[i]);
      printf("\nExecution time: %f ms", time);
      printf("\nGFlops: %f\n\n", getGflops(width, time));
    } else {
	printf ("\nFailed ! Wrong result (for C[%d][%d])\n\n", i, j);
    }
    
    free (A);
    free (B);
    free (C);

    status = cublasXtDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }
    
    return 0;
}

float getGflops (int width, float time) {

	float gf = (2.0e-6 * width * width* width / time);

	return gf;
}
