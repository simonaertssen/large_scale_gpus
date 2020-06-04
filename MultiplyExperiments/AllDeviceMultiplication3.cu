#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADS 4

/*
This script contains the same functionality as AllDeviceMultiplication2 but with a full buffer
run with
nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.cu AllDeviceMultiplication3.cu && ./a.out 1000 500
*/

void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, double* memacc );
void TileGPUAddToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, double* memacc );


int main(int argc, char* argv[]) {

    // set matrix size
    unsigned int n = 5000;
    if (argc > 1){
        n = (unsigned int)atoi(argv[1]);
        printf("matrix dimension = %lu, ", n);
        if (n > 40960 ) {
            printf("matrix dimension too large..\n");
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
            x = n/2;
            printf("block size too large, setting block size to %d..\n", x);
        }
    }

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocM1(n, n); // diagonal A matrix
	matrix *h_B = kuhdaMallocM1(n, n); // diagonal B matrix
	matrix *h_C = kuhdaMallocM(n, n); // empty C matrix

    int abc, ABC = 3; // counters to loop through matrices
    int device, devicecount = 4;
    int stream, streamsperdevice = 20; //(n/x)*(n/x);

    // parallel device warmup
    #pragma omp parallel for private(device) num_threads(devicecount)
    for (device = 0; device < devicecount; device ++) kuhdaWarmupDevice(device);
    
    printf("streamsperdevice = %d\n", streamsperdevice);
    GPUCHECK(cudaGetDeviceCount(&devicecount));
    matrix *d_All[devicecount][ABC];

    int streamcount = streamsperdevice*devicecount;
    int Q; // register quotient of n/x
    size_t availableMemory, queryMemory = (size_t) 3*x*x*sizeof(double), GBconv = 1000*1000*1000;
    cudaStream_t d_streams[streamcount];
    cublasHandle_t handles[devicecount];
    double *membuffs[devicecount][2];

    MatMulTimer timer;

    // Measure available memory and adjust x if necessary
    for (device = 0; device < devicecount; device++){
        // Get device properties to measure available memory:
        availableMemory = kuhdaAvailableMemoryOnCurrentDevice();
        printf("%4.2lf GB available on device %d, asking for %4.2lf GB..\n", (double)availableMemory/GBconv, device, (double) queryMemory/GBconv);
        if (availableMemory < queryMemory){
            // get surplus query memory and register how many more x's we will need, + 1 is neccessary for the integer division              
            Q = (int) n/x + (int)((availableMemory - queryMemory)/x) + 1; 
            printf("Q was %d, now Q is %d\n", n/x, Q);
            x = (unsigned int) n/Q;
            queryMemory = 3*x*x*sizeof(double);
            printf("Changed x to %d, now asking for %4.2lf GB..\n", x, (double) queryMemory/GBconv);
        } 
    }

    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasCreate(&handles[device])); 

        GPUCHECK(cudaHostAlloc(&membuffs[device][0], x*x*sizeof(double), cudaHostAllocPortable));
        GPUCHECK(cudaHostAlloc(&membuffs[device][1], x*x*sizeof(double), cudaHostAllocPortable));

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
        }
        
        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
        }
    }

    printf("Computation start..\n");
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
                GPUCHECK(cudaSetDevice(currentdevice));

                TileHostToGPUBuff(mtile*x, (mtile+1)*x, ktile*x, (ktile+1)*x, h_A, d_All[currentdevice][0], d_streams[streamindex], membuffs[currentdevice][0]); // Tile A
                TileHostToGPUBuff(ktile*x, (ktile+1)*x, ntile*x, (ntile+1)*x, h_B, d_All[currentdevice][1], d_streams[streamindex], membuffs[currentdevice][1]); // Tile B


                // damn man dads not sooo fast.. yet
                kuhdamm(d_All[currentdevice][0], d_All[currentdevice][1], d_All[currentdevice][2], d_streams[streamindex], handles[currentdevice]);

                // kuhdaPrintDeviceM(d_All[currentdevice][2]);

                // Get the tile back
                TileGPUAddToHostBuff(mtile*x, (mtile+1)*x, ntile*x, (ntile+1)*x, d_All[currentdevice][2], h_C, d_streams[streamindex], membuffs[currentdevice][0]);

                // Check whether current stream is available:
                streamindex++;
                streamindex = streamindex%streamcount;

                // kuhdaPrintM(h_C);
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
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasDestroy(handles[device]));

        GPUCHECK(cudaFreeHost(membuffs[device][0]));
        GPUCHECK(cudaFreeHost(membuffs[device][1]));

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
        }
        // Takes NO arguments
        GPUCHECK(cudaDeviceReset());
    }

	return 0;
}


void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, double* memacc )
{	
    // check input
    if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long cols = colstop - colstart, rows = rowstop - rowstart, i, j;

    for (i=rowstart; i<rowstop; ++i){
        for (j=colstart; j<colstop; ++j){
            // fill memacc with host-matrix data one (tile-)row at a time:
            // memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
            memacc[(i - rowstart) * h_matrix->c + (j - colstart)] = h_matrix->data[i * h_matrix->c + j];
        }
    }
    
    GPUCHECK(cudaMemcpyAsync((void*)&d_tile->data[0], (const void*)memacc, rows*cols*sizeof(double), cudaMemcpyHostToDevice, stream));
    GPUCHECK(cudaStreamSynchronize(stream));
}

void TileGPUAddToHostBuff( unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, double* memacc )
{
    // check input
    if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long cols = colstop - colstart, rows = rowstop - rowstart, i, j;

    GPUCHECK(cudaMemcpyAsync(memacc, (void*)&d_tile->data[0], rows*cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaStreamSynchronize(stream));
    for (i = rowstart; i < rowstop; ++i){
        for (j = colstart; j < colstop; ++j){
            h_matrix->data[i * h_matrix->c + j] += memacc[(i - rowstart) * h_matrix->c + (j - colstart)];
        }
    }
}
