// Program to test the allocation and sending of quarter tiles

#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADS 3

// Run with:
// nvcc -O3 lXcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.c TwoDeviceTileMultiplication.c && ./a.out

// What do we want to test: (in parallel)
// Send d_A1 and d_B1 to device 3 and d_A2 and d_B3 to device 2
// call kuhdamm() to try and compute in parallel

// Results: everything works, see how every element in the quarter tile = n

int main()
{   
    //omp_set_num_threads(2);
    //omp_set_dynamic(0);
	unsigned long n = 10000, size = n * n * sizeof(double);
	int x = n/2, sizex = x * x * sizeof(double); // x * x = dimension of quarter tile

	// Containers for host and device matrices
	matrix *h_A  = kuhdaMallocMP1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocMP1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocMP(n, n); // empty C matrix

    int abc, ABC = 3, device, devicecount = 4;
    matrix *d_All[devicecount][ABC];

    cudaStream_t d_streams[devicecount], mainstream;
    cudaEvent_t start[devicecount], stop[devicecount], mainstart, mainstop;
 
    gpuErrchk(cudaStreamCreate(&mainstream));
    gpuErrchk(cudaEventCreate(&mainstart));
	gpuErrchk(cudaEventCreate(&mainstop));

    float ms_timer[4] = {0.0, 0.0, 0.0, 0.0}, mainstreamtimer;
    #pragma omp parallel for private(abc) private(device) num_threads(NUMTHREADS)
    // Creat all dependencies:
    for (device = 2; device < devicecount; device++){
        //printf("Number of threads = %d\n", omp_get_thread_num());
        gpuErrchk(cudaSetDevice(device));
        printf("Allocating tiles A, B and C on device %d\n", device);

        gpuErrchk(cudaStreamCreate(&d_streams[device]));
        gpuErrchk(cudaEventCreate(&start[device]));
	    gpuErrchk(cudaEventCreate(&stop[device]));

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
        }

        // Send the first quarter tiles of A and B to device 0...
	    TileHostToGPU(0, x, 0, x, h_A, d_All[device][0], d_streams[device]);
	    TileHostToGPU(0, x, 0, x, h_B, d_All[device][1], d_streams[device]);
    }

    gpuErrchk(cudaStreamSynchronize(mainstream));
    gpuErrchk(cudaEventRecord(mainstart, mainstream));

    int checkCorrectDevice = 0;
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 2; device < devicecount; device++){
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaGetDevice(&checkCorrectDevice));
        if (checkCorrectDevice != device) printf("Wrong device?");

        //gpuErrchk(cudaStreamSynchronize(d_streams[device]));
	    //gpuErrchk(cudaEventRecord(start[device], d_streams[device]));

        // Matrix multiplication: damm man that's fast 
	    kuhdamm(d_All[device][0], d_All[device][1], d_All[device][2], d_streams[device], 0);

	    //gpuErrchk(cudaStreamSynchronize(d_streams[device]));
        gpuErrchk(cudaEventRecord(stop[device], d_streams[device]));
	    gpuErrchk(cudaEventSynchronize(stop[device]));

	    gpuErrchk(cudaEventElapsedTime(&ms_timer[device], start[device], stop[device]));
	    printf("Multiplication on device %d took %lf seconds\n", device, ms_timer[device]/1000);

        // ...retrieve it again into C on the host
        //TileGPUAddToHost(0, x, 0, x, d_All[device][2], h_C, d_streams[device]);
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 2; device < devicecount; device++){
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaStreamSynchronize(d_streams[device]));
        TileGPUAddToHost(0, x, 0, x, d_All[device][2], h_C, d_streams[device]);
    }

    gpuErrchk(cudaEventRecord(mainstop, mainstream));
    gpuErrchk(cudaEventSynchronize(mainstop));
    gpuErrchk(cudaEventElapsedTime(&mainstreamtimer, mainstart, mainstop));
    int timerindex;
    for (timerindex = 1; timerindex < devicecount; timerindex++) ms_timer[0] += ms_timer[timerindex];
	printf("Everything: %lf s, multiplication: = %lf s\n", mainstreamtimer/1000, ms_timer[0]/1000);

    kuhdaTestM(0, x, 0, x, h_C);
    //kuhdaPrintM(h_C);
    //printf("%lf  %lf \n%lf  %lf \n", h_C->data[(n-1)*x-1], h_C->data[(n-1)*x], h_C->data[n*x-1], h_C->data[n*x]);

    // free all matrices
    printf("Cleaning up ..\n");

    gpuErrchk(cudaStreamDestroy(mainstream));
    gpuErrchk(cudaEventDestroy(mainstart));
	gpuErrchk(cudaEventDestroy(mainstop));

	kuhdaFreeM(h_A, 'p');
	kuhdaFreeM(h_B, 'p');
	kuhdaFreeM(h_C, 'p');
    for (device = 2; device < devicecount; device++){
        gpuErrchk(cudaSetDevice(device));

        gpuErrchk(cudaStreamDestroy(d_streams[device]));
        gpuErrchk(cudaEventDestroy(start[device]));
	    gpuErrchk(cudaEventDestroy(stop[device]));

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
        }

        // Takes NO arguments
        gpuErrchk(cudaDeviceReset());
    }

	return 0;
}
