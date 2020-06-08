#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADS 4
#define NUMTHREADSBUFF 16
#define TILES 4


/*
With this script we investiate intergpu communication to speed up dgemm. It builds on the full buffer used in AllDeviceMultiplication3.
We are only investigating 4 tiles, due to easy extrapolation.

nvcc -O3 -Xcompiler -fopenmp -lcublas --default-stream per-thread -Xcompiler -mno-float128 ../DIEKUHDA/kuhda.cu AllDeviceMultiplicationPeer.cu && ./a.out 1000 500
*/

void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, matrix *memacc );
void TileGPUAddToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc );


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

    // Permutation of the devices for NVLINK exploit
    int d[4] = {0, 1, 3, 2};
    int abc, ABC = 3; // counters to loop through matrices
    int device, currentdevice, destination, devicecount = 4;
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
    cudaStream_t p_streams[TILES];
    cublasHandle_t handles[devicecount];
    matrix *membuffs[devicecount];

    MatMulTimer timer;

    // Check dimensions with regards to the available memory:
    x = kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, currentdevice, abc, stream) num_threads(NUMTHREADS)
    // Creat all dependencies:
    for (device = 0; device < devicecount; device++){
        currentdevice = d[device];
        GPUCHECK(cudaSetDevice(currentdevice));
        CUBLASCHECK(cublasCreate(&handles[currentdevice])); 

        membuffs[currentdevice] = kuhdaMallocMP(x, x);

        for (abc = 0; abc < ABC; ++abc){
            d_All[currentdevice][abc] = kuhdaMallocDeviceM(x, x);
        }
        
        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamCreate(&d_streams[currentdevice + stream*devicecount]));
        }

        // Create the P2P streams
        if (currentdevice == 0 || currentdevice == 1) {
            GPUCHECK(cudaStreamCreate(&p_streams[currentdevice]));
            GPUCHECK(cudaStreamCreate(&p_streams[currentdevice + 2]));

            // Set accessibility:
            cudaDeviceEnablePeerAccess((currentdevice + 1)%2, 0);
        }
    }

    printf("Computation start..\n");
    timer.Start();

    device = 0;
    int streamindex = 0, loopindex = 0;
    int mtile = 0, ntile = 0, ktile = 0;
    // Loop over rows of A:
    //#pragma omp parallel for private(mtile)
    for (mtile = 0; mtile < m/x; ++mtile){
        // Loop over columns of B:
        for (ntile = 0; ntile < n/x; ++ntile){
            // #pragma omp parallel for private(ktile, device, currentdevice) num_threads(NUMTHREADS)
            // Loop over columns of A and rows of B:
            for (ktile = 0; ktile < k/x; ++ktile){
                // Set device by using integer division: 0, 0, 0, 1, 1, 1, ...
                //currentdevice = streamindex/streamsperdevice;
                //printf("device %d becomes device %d\n", device, d[device]);

                currentdevice = d[device];
                GPUCHECK(cudaSetDevice(currentdevice));

                GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));
                GPUCHECK(cudaStreamSynchronize(p_streams[(currentdevice + 2)%TILES]));

                TileHostToGPUBuff(mtile*x, (mtile+1)*x, ktile*x, (ktile+1)*x, h_A, d_All[currentdevice][0], d_streams[streamindex], membuffs[currentdevice]); // Tile A
                if (n == 0 && (currentdevice == 0 || currentdevice == 1)){ // P2P copy, after device has gotten tile A
                    GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));
                    gpuErrchk(cudaMemcpyPeerAsync(d_All[currentdevice+2][0], currentdevice+2, d_All[currentdevice][0], currentdevice, x*x*sizeof(double), p_streams[currentdevice + 2*mtile]));
                }

                if (mtile < 1){
                    TileHostToGPUBuff(ktile*x, (ktile+1)*x, ntile*x, (ntile+1)*x, h_B, d_All[currentdevice][1], d_streams[streamindex], membuffs[currentdevice]); // Tile B
                    GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));
                }

                // damn man dads not sooo fast.. yet
                kuhdamm(d_All[currentdevice][0], d_All[currentdevice][1], d_All[currentdevice][2], d_streams[streamindex], handles[currentdevice]);

                // Get the tile back
                TileGPUAddToHostBuff(mtile*x, (mtile+1)*x, ntile*x, (ntile+1)*x, d_All[currentdevice][2], h_C, d_streams[streamindex], membuffs[currentdevice]);

                device++;
                if (device != 0 && device%devicecount == 0) loopindex++;
                device = device%devicecount;
                // 'device' increments 0 to 3, 'd[device]' is a permutation. 'streamindex' is based on the permutation because that was easier to implement in the allocation and destruction loops.
                streamindex = d[device] + loopindex*devicecount;
                streamindex = streamindex%streamcount;
            }
        }
    }


    timer.Stop();
    double timingResult = timer.GFLOPS_DGEMM(m, n, k);
    printf("GFLOPS = %.0lf\n", timingResult);

    // Test the result for mistakes
	kuhdaTestM(0, n, 0, n, h_C);

    // Free all
    printf("Cleaning up ..\n");
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();

    #pragma omp parallel for private(device, currentdevice, abc, stream) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
        currentdevice = d[device];
        GPUCHECK(cudaSetDevice(currentdevice));
        CUBLASCHECK(cublasDestroy(handles[currentdevice]));

        kuhdaFreeM(membuffs[currentdevice], 'p');

        for (abc = 0; abc < ABC; ++abc){
            kuhdaFreeM(d_All[currentdevice][abc], 'c');
        }

        for (stream = 0; stream < streamsperdevice; ++stream){
            GPUCHECK(cudaStreamDestroy(d_streams[currentdevice + stream*devicecount]));
        }

        if (currentdevice == 0 || currentdevice == 1) {
            GPUCHECK(cudaStreamDestroy(p_streams[currentdevice]));
            GPUCHECK(cudaStreamDestroy(p_streams[currentdevice + 2]));
        }

        // Takes NO arguments
        GPUCHECK(cudaDeviceReset());
    }

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
    
    // GPUCHECK(cudaMemcpyAsync((void*)&d_tile->data[0], (void*)&memacc->data[0], rows*cols*sizeof(double), cudaMemcpyHostToDevice, stream));
    GPUCHECK(cudaMemcpy2DAsync((void*)&d_tile->data[0], memacc->c*sizeof(double), (const void*)&memacc->data[0], memacc->c*sizeof(double),
                                memacc->c*sizeof(double), memacc->r, cudaMemcpyHostToDevice, stream));

    // GPUCHECK(cudaMemcpy2DAsync(
    // (void*)(&d_A[device]->data[0]),
    // tiledim*sizeof(double),
    // (const void*)(&h_A->data[destinations[device][0] * h_A->c + destinations[device][2]]),
    // n*sizeof(double),
    // tiledim*sizeof(double),
    // tiledim,
    // cudaMemcpyHostToDevice,
    // d_streams[device*streamsperdevice]));
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

    // GPUCHECK(cudaMemcpyAsync((void*)&memacc->data[0], (void*)&d_tile->data[0], rows*cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaMemcpy2DAsync((void*)&memacc->data[0], memacc->c*sizeof(double), (const void*)&d_tile->data[0], d_tile->c*sizeof(double),
                                d_tile->c*sizeof(double), d_tile->r, cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaStreamSynchronize(stream));

    #pragma omp parallel for private(i) num_threads(NUMTHREADSBUFF)
    for (i = rowstart; i < rowstop; ++i){
        for (j = colstart; j < colstop; ++j){
            h_matrix->data[i * h_matrix->c + j] += memacc->data[(i - rowstart) * memacc->c + (j - colstart)];
        }
    }
}
