// This file contains some of the kernel functions necessary to test matmulttes
#include "cuda.h"
#include "cuda_runtime.h"

extern "C" {
	#include "kernels.h"
}
#include <stdio.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void hello_device(int printme) { 
    // C doens't work here?
    //printf("Device 0 says hello to integer %d\n", printme); 
    //std::cout << "Hello from device 0 to integer" << printme << std::endl;
    //printf("Hello thread %d, integer = %d\n", threadIdx.x, printme);

}

extern "C" void hello_device_wrapper(int printme){
    printf("Device 0 says hello to integer %d\n", printme); 
    hello_device <<<10,10>>>(printme);
}

__global__ void gpu_mul(double const * const A, double * const C, const int rows_A, const int cols_A) {
	//printf("Hello from the device\n");
	//Each Thread computes one element of C
	double C_element = 0.0;
	const int 	row = blockIdx.y * blockDim.y + threadIdx.y,
				col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	if (row < cols_A && col < cols_A) 
	{
		for (k = 0; k < rows_A; ++k) {
			C_element += A[k * cols_A + row] * A[k * cols_A + col];
		}
		C[row * cols_A + col] = C_element;
	}
}

extern "C" void gpu_mul_wrapper(double const * const A, double * const C, const int rows_A, const int cols_A){
    gpu_mul<<<1, 1>>>(A, C, rows_A, cols_A);
}

/*int main(){
    int printme = 12;
    hello_device_wrapper(printme);
    return 0;
}*/