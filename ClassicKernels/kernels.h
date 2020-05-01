#ifndef kernels
#define kernels

__global__ void hello_device(int printme);
extern "C" void hello_device_wrapper(int printme);

__global__ void gpu_mul(double const * const A, double * const C, const int rows_A, const int cols_A);
extern "C" void gpu_mul_wrapper(double const * const A, double * const C, const int rows_A, const int cols_A); 

#endif