#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADSBUFF 16

/*
This script builds on AllDeviceMultiplication3.cu, but takes into account the comments of HH (11/06/2020).
Full concurrency was previously not attained due to failing parallellism in the main for loops. Here, we adjust for that concurrency.
This is the 'naive' implementation of multi-gpu computing: only H2D and D2H comms.

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

    // Set matrix size
    unsigned int n = 5000;
    if (argc > 1){
        n = (unsigned int)atoi(argv[1]);
        if (n >= 73029) {
            // In this case, n exceeds the size necessary for 128 GB of data on the host for three matrices: sqrt(128 * pow(1000, 3) / 8 / 3) = 73029
            printf("Matrix dimension too large, setting matrix size to %d..\n", n);
            return -1;
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
    int device, devicecount;
    GPUCHECK(cudaGetDeviceCount(&devicecount));
    x = kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocM1(n, n); // matrix A filled with ones
	matrix *h_B = kuhdaMallocM1(n, n); // matrix B filled with ones
    matrix *h_C = kuhdaMallocM(n, n);  // matrix C will contain results: n in every spot due to type of A and B
    int abc, ABC = 3; 
    matrix *d_All[devicecount][ABC];   // matrix tiles on each device

    // Counters for streams
    int stream, streamsperdevice = (int) pow(2, (int) n/x);
    streamsperdevice = streamsperdevice > 32 ? 32 : streamsperdevice;
    int streamcount = streamsperdevice*devicecount;

    // Parallel device warmup
    #pragma omp parallel for private(device) num_threads(devicecount)
    for (device = 0; device < devicecount; device ++) kuhdaWarmupDevice(device);
    
    // Cuda dependencies
    cudaStream_t d_streams[streamcount];
    cublasHandle_t handles[devicecount];
    matrix *membuffs[devicecount];

    MatMulTimer timer;

    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasCreate(&handles[device])); 

        membuffs[device] = kuhdaMallocMP(x, x);

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
        }
        
        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamCreate(&d_streams[device + stream*devicecount]));
        }
    }

    printf("Computation start..\n");
    timer.Start();

    int streamindex = 0, currentdevice = 0, loopindex = 0;
    int mtile = 0, ntile = 0, ktile = 0;
    // Loop over rows of A:
    //#pragma omp parallel for private(mtile)
    for (mtile = 10; mtile < m/x; ++mtile){
        // Loop over columns of B:
        for (ntile = 10; ntile < n/x; ++ntile){
            // #pragma omp parallel for private(ktile) num_threads(NUMTHREADS)
            // Loop over columns of A and rows of B:
            for (ktile = 10; ktile < k/x; ++ktile){
                // Set device by using integer division: 0, 0, 0, 1, 1, 1, ...
                //currentdevice = streamindex/streamsperdevice;
                GPUCHECK(cudaSetDevice(currentdevice));

                TileHostToGPUBuff(mtile*x, (mtile+1)*x, ktile*x, (ktile+1)*x, h_A, d_All[currentdevice][0], d_streams[streamindex], membuffs[currentdevice]); // Tile A
                TileHostToGPUBuff(ktile*x, (ktile+1)*x, ntile*x, (ntile+1)*x, h_B, d_All[currentdevice][1], d_streams[streamindex], membuffs[currentdevice]); // Tile B

                // We are using two different streams to try out
                GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));

                // damn man dads not sooo fast.. yet
                kuhdamm(d_All[currentdevice][0], d_All[currentdevice][1], d_All[currentdevice][2], d_streams[streamindex], handles[currentdevice]);

                // Get the tile back
                TileGPUAddToHostBuff(mtile*x, (mtile+1)*x, ntile*x, (ntile+1)*x, d_All[currentdevice][2], h_C, d_streams[streamindex], membuffs[currentdevice]);

                currentdevice++;
                if (currentdevice != 0 && currentdevice%devicecount == 0) loopindex++;
                currentdevice = currentdevice%devicecount;
                streamindex = currentdevice + loopindex*devicecount;
                streamindex = streamindex%streamcount;
            }
        }
    }

    // Final synchronization:
    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    for (device = 0; device < devicecount; device ++){
        cudaSetDevice(device);
        cudaDeviceSynchronize();
      }

    timer.Stop();
    double timingResult = timer.GFLOPS_DGEMM(m, n, k);
    printf("GFLOPS = %.0lf\n", timingResult);

    // Test the result for mistakes
	// kuhdaTestM(0, n, 0, n, h_C);

    // Free all dependencies
    printf("Cleaning up ..\n");
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasDestroy(handles[device]));

        kuhdaFreeM(membuffs[device], 'p');

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamDestroy(d_streams[device + stream*devicecount]));
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

    #pragma omp parallel for private(i) num_threads(NUMTHREADSBUFF)
    for (i=rowstart; i<rowstop; ++i){
        for (j=colstart; j<colstop; ++j){
            // fill memacc with host-matrix data one (tile-)row at a time:
            // memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
            memacc->data[(i - rowstart) * memacc->c + (j - colstart)] = h_matrix->data[i * h_matrix->c + j];
        }
    }
    
    GPUCHECK(cudaMemcpyAsync((void*)&d_tile->data[0], (void*)&memacc->data[0], rows*cols*sizeof(double), cudaMemcpyHostToDevice, stream));
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

    #pragma omp parallel for private(i) num_threads(NUMTHREADSBUFF)
    for (i = rowstart; i < rowstop; ++i){
        for (j = colstart; j < colstop; ++j){
            h_matrix->data[i * h_matrix->c + j] += memacc->data[(i - rowstart) * memacc->c + (j - colstart)];
        }
    }
}
