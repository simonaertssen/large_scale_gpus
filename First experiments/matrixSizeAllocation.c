#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"


int main(){
    // Get the desired size:
    double *data = NULL;
    unsigned long long n = 60000;
    unsigned long long memsize = n*n*sizeof(double);
    long double printmemsize = memsize / 1.0e9;

    //printf("memsize = %.3Lf Gb\n", printmemsize);
    int failure = gpuErrchk(cudaMalloc((void**)&data, memsize)); // Tip from HH
    //printf("Failure = %d\n", failure);

    // Read device info:
    int device, devicecount;
	struct cudaDeviceProp prop;
	gpuErrchk(cudaGetDeviceCount(&devicecount));
    long double totalMem, totalMemGb, totalMemGB;

    for (device = 0; device < devicecount; ++device){
		// Set the current device:
		gpuErrchk(cudaSetDevice(device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		printf("Device %s: %d of %d\n", prop.name, device, devicecount);
        totalMem = prop.totalGlobalMem;
        totalMemGb = totalMem / 1.0e9;
        totalMemGB = totalMem / (1024*1024*1024);
        printf("Global memory on device = %.0Lf bytes = %.3Lf Gb = %.3Lf GB\n", totalMem, totalMemGb, totalMemGB);
    }
    
    long double bytes
    printf()


    return 0;
}