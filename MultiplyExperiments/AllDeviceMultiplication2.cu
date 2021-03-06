#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"
#include "math.h"

#define NUMTHREADS 4
#define NUMTHREADSBUFF 16

/*
With this script we are following some of the ideas from Jochen Kreuz to perform a tiled multiplication using cublas.
We use a biffer that loads and copies tiles line by line. Problems were solved by stream synchronisation inside the 
Tile...To... functions inthis script.
This is the second iteration of the algorithm, after commentary from HH. This includes:
- Perform kuhdaWarmup
- Create a cublas handle on every device in the first loop to relieve stress from kuhdamm

- Check whether some matrices need to be repeated?

run with
nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.cu AllDeviceMultiplication2.cu && ./a.out 1000 500
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
            return -1;
        }
    }

	// Containers for host and device matrices
	matrix *h_A = kuhdaMallocM1(n, n); // diagonal A matrix
	matrix *h_B = kuhdaMallocM1(n, n); // diagonal B matrix
	matrix *h_C = kuhdaMallocM(n, n); // empty C matrix

    int abc, ABC = 3; // counters to loop through matrices
    int device, devicecount = 4;
    int stream, streamsperdevice = (int) pow(2, (int) n/x);

    /* The number of streams can be computed as:
    n/x = 1:  1 streams per device, 1 loop    2**1 = 2
    n/x = 2:  2 streams per device, 8 loops   2**2 = 4
    n/x = 3:  7 streams per device, 27 loops  2**3 = 8
    n/x = 4: 16 streams per device, 64 loops  2**4 = 16
    n/x = 5: 32 streams per device, 125 loops 2**5 = 32
    Take a maximum of 32.
    */
    streamsperdevice = streamsperdevice > 32 ? 32 : streamsperdevice;

    // parallel device warmup
    #pragma omp parallel for private(device) num_threads(devicecount)
    for (device = 0; device < devicecount; device ++) kuhdaWarmupDevice(device);
    
    printf("streamsperdevice = %d\n", streamsperdevice);
    GPUCHECK(cudaGetDeviceCount(&devicecount));
    matrix *d_All[devicecount][ABC];

    int streamcount = streamsperdevice*devicecount;
    cudaStream_t d_streams[streamcount];
    cublasHandle_t handles[devicecount];
    double *membuffs[devicecount][ABC];

    MatMulTimer timer;

    // Check dimensions with regards to the available memory:
    x = kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasCreate(&handles[device])); 

        for (abc = 0; abc < ABC; ++abc){
            d_All[device][abc] = kuhdaMallocDeviceM(x, x);
            // GPUCHECK(cudaMallocHost(&membuffs[device][abc], x*sizeof(double)));
            GPUCHECK(cudaHostAlloc(&membuffs[device][abc], x*sizeof(double), cudaHostAllocPortable));
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            // GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
            GPUCHECK(cudaStreamCreate(&d_streams[device + stream*devicecount]));
            // printf("Streamindices = %d\n", device + stream*devicecount);
        }
    }

    printf("Computation start..\n");
    timer.Start();

    int streamindex = 0, currentdevice = 0, loopindex = 0;
    int mtile = 0, ntile = 0, ktile = 0;
    // Loop over rows of A:
    //#pragma omp parallel for private(mtile)
    for (mtile = 0; mtile < m/x; ++mtile){
        // Loop over columns of B:
        for (ntile = 0; ntile < n/x; ++ntile){
            // Loop over columns of A and rows of B:
            // #pragma omp parallel for private(ktile) private(streamindex) num_threads(NUMTHREADS)
            for (ktile = 0; ktile < k/x; ++ktile){
                // Set device by using integer division: 0, 1, 1, 3, 0, 1, ...
                
                // Was: currentdevice = streamindex/streamsperdevice;
                GPUCHECK(cudaSetDevice(currentdevice));
                // printf("Device = %d and stream = %d\n", currentdevice, streamindex);

                TileHostToGPUBuff(mtile*x, (mtile+1)*x, ktile*x, (ktile+1)*x, h_A, d_All[currentdevice][0], d_streams[streamindex], membuffs[currentdevice][0]); // Tile A
                TileHostToGPUBuff(ktile*x, (ktile+1)*x, ntile*x, (ntile+1)*x, h_B, d_All[currentdevice][1], d_streams[streamindex], membuffs[currentdevice][1]); // Tile B

                // damn man dads not sooo fast.. yet
                kuhdamm(d_All[currentdevice][0], d_All[currentdevice][1], d_All[currentdevice][2], d_streams[streamindex], handles[currentdevice]);

                // Get the tile back
                TileGPUAddToHostBuff(mtile*x, (mtile+1)*x, ntile*x, (ntile+1)*x, d_All[currentdevice][2], h_C, d_streams[streamindex], membuffs[currentdevice][2]);
                GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));

                // Check whether current stream is available:
                // streamindex++;
                // streamindex = streamindex%streamcount;

                currentdevice++;
                if (currentdevice != 0 && currentdevice%devicecount == 0) loopindex++;
                currentdevice = currentdevice%devicecount;
                streamindex = currentdevice + loopindex*devicecount;
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
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, abc, stream) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));
        CUBLASCHECK(cublasDestroy(handles[device]));

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[device][abc], 'c');
            GPUCHECK(cudaFreeHost(membuffs[device][abc]));
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            // GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
            GPUCHECK(cudaStreamDestroy(d_streams[device + stream*devicecount]));
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
    if (h_matrix == NULL || d_tile == NULL) 	INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long cols = colstop - colstart, i, j;

    // 'strided' copy, row by row
    for (i=rowstart; i<rowstop; ++i){
        #pragma omp parallel for private(j) num_threads(NUMTHREADSBUFF)
        for (j=colstart; j<colstop; ++j){
            // fill memacc with host-matrix data one (tile-)row at a time:
            memacc[j-colstart] = h_matrix->data[i * h_matrix->c + j];
        }
        GPUCHECK(cudaMemcpyAsync((void*) (&d_tile->data[0] + (cols * (i-rowstart))), memacc, cols*sizeof(double), cudaMemcpyHostToDevice, stream));
        GPUCHECK(cudaStreamSynchronize(stream));
    }
    return;
}

void TileGPUAddToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, double* memacc )
{
    // check input
    if (h_matrix == NULL || d_tile == NULL) 	INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;


    unsigned long cols = colstop - colstart, i, j;

    // 'strided' copy, row by row
    for (i=rowstart; i<rowstop; ++i){
        // takes (d_arr, h_arr, nbytes, cudaMemcpyHostToDevice, stream)
        GPUCHECK(cudaMemcpyAsync(memacc, (void*) (&d_tile->data[0] + (cols * (i-rowstart))), cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
        // failure = GPUCHECK(cudaMemcpy(memacc, (void*) (&d_tile->data[0] + (cols * (i-rowstart))), cols*sizeof(double), cudaMemcpyDeviceToHost));
        GPUCHECK(cudaStreamSynchronize(stream));
        #pragma omp parallel for private(j) num_threads(NUMTHREADSBUFF)
        for (j=colstart; j<colstop; ++j){
            h_matrix->data[i * h_matrix->c + j] += memacc[j-colstart];
        }

    }
    return;
}