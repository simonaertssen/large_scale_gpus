#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADS 4

/*
With this script we are following some of the ideas from Jochen Kreuz to perform a tiled multiplication using cublas.
This is the second iteration of the algorithm, after commentary from HH. This includes:
- Perform kuhdaWarmup
- Create a cublas handle on every device in the first loop to relieve stress from kuhdamm
- Check whether some matrices need to be repeated

run with
nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.cu AllDeviceMultiplication2.cu && ./a.out 1000 500
*/


int main(int argc, char* argv[]) {

    // set matrix size
    unsigned int n = 5000;
    if (argc > 1){
        n = (unsigned int)atoi(argv[1]);
        printf("matrix dimension = %lu\n", n);
        if (n > 40960 ) {
            printf("matrix dimension too large ..\n");
            return -1;
        }
    }
    unsigned int m = n, k = n;

    // set tile size
    unsigned int x = n/2;
    if (argc > 2){
        x = (unsigned int)atoi(argv[2]);
        printf("block size = %lu\n", x);
        if (x > n ) {
            printf("block size too large ..\n");
            return -1;
        }
    }

	// Containers for host and device matrices
	matrix *h_A  = kuhdaMallocM1(n, n); // diagonal A matrix
	matrix *h_B  = kuhdaMallocM1(n, n); // diagonal B matrix
	matrix *h_C  = kuhdaMallocM(n, n); // empty C matrix

    int abc, ABC = 3; // counters to loop through matrices
    int device, devicecount = 4;
    int stream, streamsperdevice = 20; //(n/x)*(n/x);
    
    printf("streamsperdevice = %d\n", streamsperdevice);
    gpuErrchk(cudaGetDeviceCount(&devicecount));
    matrix *d_All[devicecount][ABC];

    int streamcount = streamsperdevice*devicecount;
    cudaStream_t d_streams[streamcount];


    MatMulTimer timer;

    printf("Allocating tiles A, B and C on %d devices\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        //printf("Number of threads = %d\n", omp_get_thread_num());
        gpuErrchk(cudaSetDevice(device));

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            gpuErrchk(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
            //gpuErrchk(cudaEventCreate(&bufferIsFull[stream + streamsperdevice*device]));

        }
    }


    printf("Computation start\n");
    timer.Start();

    int streamindex = 0, currentdevice = 0;
    int mtile = 0, ntile = 0, ktile = 0;
    // Loop over rows of A:
    //#pragma omp parallel for private(mtile)
    for (mtile = 0; mtile < m/x; ++mtile){
        // Loop over columns of B:
        for (ntile = 0; ntile < n/x; ++ntile){
            // #pragma omp parallel for private(ktile) num_threads(NUMTHREADS)
            // Loop over columns of A and rows of B:
            for (ktile = 0; ktile < k/x; ++ktile){
                // Set device by using integer division: 0, 0, 0, 1, 1, 1, ...
                currentdevice = streamindex/streamsperdevice;
                gpuErrchk(cudaSetDevice(currentdevice));

                // Check whether current stream is available:
                gpuErrchk(cudaStreamSynchronize(d_streams[streamindex]));
                

                TileHostToGPU(mtile*x, (mtile+1)*x, ktile*x, (ktile+1)*x, h_A, d_All[currentdevice][0], d_streams[streamindex]); // Tile A
                TileHostToGPU(ktile*x, (ktile+1)*x, ntile*x, (ntile+1)*x, h_B, d_All[currentdevice][1], d_streams[streamindex]); // Tile B

                // damn man dads fast
                kuhdamm(d_All[currentdevice][0], d_All[currentdevice][1], d_All[currentdevice][2], d_streams[streamindex], 0);

                gpuErrchk(cudaDeviceSynchronize());
                // Get the tile back
                TileGPUAddToHost(mtile*x, (mtile+1)*x, ntile*x, (ntile+1)*x, d_All[currentdevice][2], h_C, d_streams[streamindex]); // credzz to louis

                streamindex++;
                streamindex = streamindex%streamcount;
            }
        }
    }


    timer.Stop();
    double timingResult = timer.GFLOPS_DGEMM(m, n, k);
    printf("GFLOPS = %.2lf\n", timingResult);

    //h_C->data[100] = 578.0;
    // Test the result for mistakes
	kuhdaTestM(0, n, 0, n, h_C);
    //printf("%lf  %lf \n%lf  %lf \n", h_C->data[(n-1)*x-1], h_C->data[(n-1)*x], h_C->data[n*x-1], h_C->data[n*x]);

    // Free all
    printf("Cleaning up ..\n");
    gpuErrchk(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
        gpuErrchk(cudaSetDevice(device));

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            gpuErrchk(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
        }
        // Takes NO arguments
        gpuErrchk(cudaDeviceReset());
    }

	return 0;
}
