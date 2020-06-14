#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADSBUFF 16

/*
This script builds on AllDeviceMultiplication3.cu, but takes into account the comments of HH (11/06/2020).
Full concurrency was previously not attained due to failing parallellism in the main for loops. Here, we adjust for that concurrency.
This is the 'naive' implementation of multi-gpu computing: only H2D and D2H comms, and only up to four tiles.

The challenge here is that different tiles of C are overwritten in TileGPUAddToHostBuff at the same time, so even though we have a pragma omp 
parallel for loop to run all devices in parallel in such way that each device can recycle tiles of B, we need to synchronize on the tiles of C.

run with
nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.cu ADM_NaiveBuff.cu && ./a.out 1000 500
*/

void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, matrix *memacc );
void TileGPUAddToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc );


int main(int argc, char* argv[]) {
    // Prepare timer for full length of this script:
    double start = omp_get_wtime(), end; 

    // Parallel device warmup by handle creation instead of kuhdaWarmupDevice(device);
    int device, devicecount = 4;

    cublasHandle_t handles[devicecount]; 
    #pragma omp parallel for private(device) num_threads(devicecount)
    for (device = 0; device < devicecount; device ++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasCreate(&handles[device]));
    }      

    // Set matrix size
    unsigned int n = 5000;
    if (argc > 1){
        n = (unsigned int)atoi(argv[1]);
        if (n >= 73029) {
            n = 73029;
            // In this case, n exceeds the size necessary for 128 GB of data on the host for three matrices: sqrt(128 * pow(1000, 3) / 8 / 3) = 73029
            printf("Matrix dimension too large, setting matrix size to %d..\n", n);
        }
    }
    unsigned int m = n, k = n;

    // Set tile size
    unsigned int x = n/2;
    if (argc > 2){
        x = (unsigned int)atoi(argv[2]);
        if (x > n ) {
            x = n/2;
            printf("Block size too large, setting block size to %d..\n", x);
        } else if (x >= 63245){
            // In this case, x exceeds the size necessary for 32 GB of data on a device: sqrt(32 * pow(1000, 3) / 8) = 63245
            x /= 2;
        }
    }    
    printf("Matrix dimension = %lu, block size = %lu.. \n", n, x);

    // Check dimensions with regards to the available memory:
    x = kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocMdiag(n, n); // matrix A as a diagonal matrix
    matrix *h_B = kuhdaMallocM(n, n);     // matrix B to be filled with specific values for specific testing
    matrix *h_C = kuhdaMallocM(n, n);     // matrix C will contain results: same values at each spot as in b

    unsigned long i, j;
    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF)
	for (i = 0; i < h_B->r; ++i){
		for (j = 0; j < h_B->c; ++j){
            h_B->data[i*h_B->c + j] = (i + j) * 0.1 + i;
        }
    }
    
    int abc, ABC = 3; 
    matrix *d_All[devicecount][ABC];   // matrix tiles on each device

    // For numtiles tiles in each dimension, we have pow(numtiles, 3) number of operations on tiles;
    int numtiles = n/x, numtileops = pow(numtiles, 3), numtileopsperdevice = max(1, (int)ceil(numtileops/devicecount));

    // Counters for streams: number of streams is number of operations per device
    int maxstreamsperdevice = 32, stream, streamsperdevice = numtileopsperdevice;
    streamsperdevice = streamsperdevice > maxstreamsperdevice ? maxstreamsperdevice : streamsperdevice;
    int streamcount = streamsperdevice*devicecount;
    
    // Cuda dependencies
    cudaStream_t d_streams[streamcount];
    cudaEvent_t deviceBusy[devicecount];
    cudaEvent_t deviceReady[devicecount];
    matrix *membuffs[devicecount];

    MatMulTimer timer;

    // Parallel device memory and dependency allocation
    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        membuffs[device] = kuhdaMallocMP(x, x);

        GPUCHECK(cudaEventCreate(&deviceReady[device]));
        GPUCHECK(cudaEventCreate(&deviceBusy[device]));

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
        }
        
        for (stream = 0; stream < streamsperdevice; ++stream){
            // Device O contains streams 0, 1, 2, ..., device 1 contains streams streamsperdevice, streamsperdevice + 1, ... 
            GPUCHECK(cudaStreamCreate(&d_streams[device*streamsperdevice + stream]));
        }
    }

    // Main loop counters:
    int streamindex, tileopondevice, Arow, Acol, Brow, Bcol, Crow, Ccol;

    printf("Computation start..\n");
    timer.Start();

    // Parallel device multiplication loop
    #pragma omp parallel for private(device, streamindex, tileopondevice, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(devicecount)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));

        // Count what tile operation we are currently dealing with
        for (tileopondevice = 0; tileopondevice < numtileopsperdevice; tileopondevice++){
            GPUCHECK(cudaEventSynchronize(deviceReady[device]));
            streamindex = (device*streamsperdevice + tileopondevice)%streamcount;

            Arow = tileopondevice; Acol = device%2; Brow = device%2; Bcol = device/2; Crow = tileopondevice; Ccol = device/2; 
            // printf("device %d: A (%d, %d) and B (%d, %d) and C (%d, %d)\n", device, Arow, Acol, Brow, Bcol, Crow, Ccol);

            TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][0], d_streams[streamindex], membuffs[device]); // Tile A
            if (tileopondevice == 0) TileHostToGPUBuff(Brow*x, (Brow+1)*x, Bcol*x, (Bcol+1)*x, h_B, d_All[device][1], d_streams[streamindex], membuffs[device]); // Tile B

            // damn man dads not sooo fast.. yet
            kuhdamm(d_All[device][0], d_All[device][1], d_All[device][2], d_streams[streamindex], handles[device]);
            GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));

            // Get the tile back
            // printf("device %d: streamindex = %d\n", device, streamindex);
            if (device%2 == 0){ 
                GPUCHECK(cudaEventSynchronize(deviceReady[device+1]));
                // GPUCHECK(cudaStreamWaitEvent(d_streams[streamindex + 2], deviceReady[device + 1], 0));
                // Synchronise on the streams accessing the same tile of C: streams 0 and 2 access the same elements, but they are on devices 0 and 1 respecively.
                // GPUCHECK(cudaStreamSynchronize(d_streams[(streamindex + 2)%streamcount]));
                // printf("device %d: streamindex = %d synchronizes on streamindex = %d\n", device, streamindex, (streamindex + 2)%streamcount);
            } else {
                GPUCHECK(cudaEventSynchronize(deviceReady[device-1]));
                // GPUCHECK(cudaStreamWaitEvent(d_streams[streamindex - 2], deviceReady[device - 1], 0));
                // GPUCHECK(cudaStreamSynchronize(d_streams[(streamindex - 2)%streamcount]));
                // printf("device %d: streamindex = %d synchronizes on streamindex = %d\n", device, streamindex, (streamindex - 2)%streamcount);
            }

            // Record event that one stream has reached this place
            GPUCHECK(cudaEventRecord(deviceBusy[device], d_streams[streamindex]));
            TileGPUAddToHostBuff(Crow*x, (Crow+1)*x, Ccol*x, (Ccol+1)*x, d_All[device][2], h_C, d_streams[streamindex], membuffs[device]);
            GPUCHECK(cudaEventRecord(deviceReady[device], d_streams[streamindex]));
        }
        cudaDeviceSynchronize();
    }

    timer.Stop();
    double timingResult = timer.GFLOPS_DGEMM(m, n, k);
    printf("GFLOPS = %.0lf..\n", timingResult);

    // Test the result for mistakes
    printf("Checking results. ");
    double abserror = 0.0, totalerror = 0.0;
    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF) reduction(+:totalerror)
	for (i = 0; i < h_B->r; ++i){
		for (j = 0; j < h_B->c; ++j){
            abserror = fabs(h_B->data[i*h_B->c + j] - h_C->data[i*h_C->c + j]);
            totalerror += abserror;
            if (abserror > 10e-6) {
                // printf("Failure: B[%d] = %1.4e != C[%d] = %1.4e\n", i*h_B->c + j, h_B->data[i*h_B->c + j], i*h_C->c + j, h_C->data[i*h_C->c + j]);
                break;
            }
        }
    }
    if (totalerror < 10e-6) printf("Succes");
    printf("Total error of %6.2e..\n", totalerror);

    // Free all dependencies
    printf("Cleaning up..\n");
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasDestroy(handles[device]));

        GPUCHECK(cudaEventDestroy(deviceReady[device]));
        GPUCHECK(cudaEventDestroy(deviceBusy[device]));

        kuhdaFreeM(membuffs[device], 'p');

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamDestroy(d_streams[device*streamsperdevice + stream]));
        }
        // Takes NO arguments
        GPUCHECK(cudaDeviceReset());
    }

    end = omp_get_wtime(); 
    printf("Script took %.1f seconds.. \n", end - start);
	return 0;
}


void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, matrix *memacc )
{	
    // check input
    if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long cols = colstop - colstart, rows = rowstop - rowstart, i, j;

    #pragma omp parallel for private(i, j) num_threads(NUMTHREADSBUFF)
    for (i=rowstart; i<rowstop; ++i){
        for (j=colstart; j<colstop; ++j){
            memacc->data[(i - rowstart) * memacc->c + (j - colstart)] = h_matrix->data[i * h_matrix->c + j];
        }
    }
    
    GPUCHECK(cudaMemcpyAsync((void*)&d_tile->data[0], (void*)&memacc->data[0], rows*cols*sizeof(double), cudaMemcpyHostToDevice, stream));
    GPUCHECK(cudaStreamSynchronize(stream));
}

void TileGPUAddToHostBuff( unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc )
{
    // check input
    if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long cols = colstop - colstart, rows = rowstop - rowstart, i, j;

    GPUCHECK(cudaMemcpyAsync((void*)&memacc->data[0], (void*)&d_tile->data[0], rows*cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaStreamSynchronize(stream));

    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF)
    for (i = rowstart; i < rowstop; ++i){
        for (j = colstart; j < colstop; ++j){
            // #pragma omp atomic
            h_matrix->data[i * h_matrix->c + j] += memacc->data[(i - rowstart) * memacc->c + (j - colstart)];
        }
    }
}
