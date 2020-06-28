#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADSBUFF 16
#define MAXSTREAMSPERD 4
#define A 0
#define B 1
#define C 2


/*
This script builds on ADM_Naive.cu, but takes into account the functionality of BLASX with statis job scheduling.
Each device is associated with different tiles of C, and each device only computes it's own tiles of C (no computational overlap and no synchronisation between devices).
All jobs are statically scheduled: one for loop over the devices, one for loop for every of the four streams on the device. 

run with
nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.cu ADM_Direct.cu && ./a.out 1000 
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
            n = 73028;
            // In this case, n exceeds the size necessary for 128 GB of data on the host for three matrices: sqrt(128 * pow(1000, 3) / 8 / 3) = 73029
            printf("Matrix dimension too large, setting matrix size to %d..\n", n);
        }
    }
    unsigned int m = n, k = n;

    // Set tile size
    unsigned int x = n/8;
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
    int tileop, numtilesperdim = n/x, numtilestotal = numtilesperdim*numtilesperdim, numtilesperdev = numtilestotal/devicecount, streamop, numtilesperstream = numtilesperdev/MAXSTREAMSPERD;
    printf("numtilestotal = %d\n", numtilestotal);
    printf("numtilesperdev = %d\n", numtilesperdev);
    printf("numtilesperstream = %d\n", numtilesperstream);


    // Check dimensions with regards to the available memory:
    int testx = kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

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
    matrix *d_All[devicecount][ABC][MAXSTREAMSPERD];   // matrix tiles on each device per stream

    // Counters for streams: number of streams is number of operations per device
    int stream, streamcount = MAXSTREAMSPERD*devicecount;
    
    // Cuda dependencies
    cudaStream_t d_streams[devicecount][MAXSTREAMSPERD];
    matrix *membuffs[devicecount][MAXSTREAMSPERD];

    MatMulTimer timer;

    // Parallel device memory and dependency allocation
    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    // Creat all dependencies:
    for (device = 0; device < devicecount; ++device){
        GPUCHECK(cudaSetDevice(device));

        #pragma omp parallel for private(stream, abc) num_threads(MAXSTREAMSPERD)
        for (stream = 0; stream < MAXSTREAMSPERD; ++stream){
            for (abc = 0; abc < ABC; ++abc){
                d_All[device][abc][stream] = kuhdaMallocDeviceM(x, x);
            }
            GPUCHECK(cudaStreamCreate(&d_streams[device][stream]));
            membuffs[device][stream] = kuhdaMallocMP(x, x);
        }
    }

    // Main loop counters:
    // int streamindex, tileopondevice, Arow, Acol, Brow, Bcol, Crow, Ccol;
    int Arow, Acol, Brow, Bcol, Crow, Ccol, tileindex;

    printf("Computation start..\n");
    timer.Start();

    // Testing out conversion from linear to bidimensional indexing:
    int testi, testj, indexi, indexj;
    for (testi = 0; testi < numtilesperdim; ++testi){
        for (testj = 0; testj < numtilesperdim; ++testj){
            indexi = testi*numtilesperdim + testj;
            printf("%3.0d", indexi);
        }
        printf("\n");
    }

    for (testi = 0; testi < numtilestotal; ++testi){
        indexi = testi/numtilesperdim;
        indexj = testi%numtilesperdim;
        if (indexj == 0) printf("\n");
        printf("(%d, %d) ", indexi, indexj);
    }
    printf("\n");


    // Parallel device multiplication loop
    // #pragma omp parallel for private(device, stream) num_threads(devicecount)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));

        // Loop over streams per device
        for (stream = 0; stream < MAXSTREAMSPERD; ++stream){
            // Loop over all operations on C per stream
            for (streamop = 0; streamop < numtilesperstream; ++streamop){
                tileindex = (device*MAXSTREAMSPERD + stream)*numtilesperstream + streamop; 
                
                Crow = tileindex/numtilesperdim; Ccol = tileindex%numtilesperdim;
                // printf("Dev %d: tileindex = %d, (%d,%d)\n", device, tileindex, Crow, Ccol);

                // Loop over all tiles of A and B to copy: Arow = Crow and Bcol = Ccol
                for (tileop = 0; tileop < numtilesperdim; ++tileop){
                    Arow = Crow;   Acol = tileop;
                    Brow = tileop; Bcol = Ccol;

                    TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][A][stream], d_streams[device][stream], membuffs[device][stream]); // Tile A
                    TileHostToGPUBuff(Brow*x, (Brow+1)*x, Bcol*x, (Bcol+1)*x, h_B, d_All[device][B][stream], d_streams[device][stream], membuffs[device][stream]); // Tile B

                    // Copy tile B into host C
                    TileGPUAddToHostBuff(Crow*x, (Crow+1)*x, Ccol*x, (Ccol+1)*x, d_All[device][2], h_C, d_streams[streamindex], membuffs[device]);
                    GPUCHECK(cudaStreamSynchronize(d_streams[device][stream]));
                }
            }
        }


        // Count what tile operation we are currently dealing with
        // for (tileopondevice = 0; tileopondevice < numtileopsperdevice; tileopondevice++){
        //     streamindex = (device*streamsperdevice + tileopondevice)%streamcount;

        //     Arow = device/2; Acol = tileopondevice; Brow = tileopondevice; Bcol = device%2; Crow = device/2; Ccol = device%2;
        //     // printf("device %d: A (%d, %d) and B (%d, %d) and C (%d, %d)\n", device, Arow, Acol, Brow, Bcol, Crow, Ccol);

        //     TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][0], d_streams[streamindex], membuffs[device]); // Tile A
        //     TileHostToGPUBuff(Brow*x, (Brow+1)*x, Bcol*x, (Bcol+1)*x, h_B, d_All[device][1], d_streams[streamindex], membuffs[device]); // Tile B

        //     // damn man dads not sooo fast.. yet
        //     kuhdamm(d_All[device][0], d_All[device][1], d_All[device][2], d_streams[streamindex], handles[device]);
        //     GPUCHECK(cudaStreamSynchronize(d_streams[streamindex]));

        //     // Get the tile back
        //     TileGPUAddToHostBuff(Crow*x, (Crow+1)*x, Ccol*x, (Ccol+1)*x, d_All[device][2], h_C, d_streams[streamindex], membuffs[device]);
        // }
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
    if (totalerror < 10e-6) printf("Succes. ");
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

        #pragma omp parallel for private(stream, abc) num_threads(MAXSTREAMSPERD)
        for (stream = 0; stream < MAXSTREAMSPERD; ++stream){
            for (abc = 0; abc < ABC; ++abc){
                kuhdaFreeM(d_All[device][abc][stream], 'c');
            }
            GPUCHECK(cudaStreamDestroy(d_streams[device][stream]));
            kuhdaFreeM(membuffs[device][stream], 'p');
        }
        // Takes NO arguments
        GPUCHECK(cudaDeviceReset());
    }

    end = omp_get_wtime(); 
    printf("Script took %.1f seconds.. \n", end - start);
	return 0;
}


void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, matrix *memacc ) {	
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

void TileGPUToHostBuff( unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc ){
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
            h_matrix->data[i * h_matrix->c + j] = memacc->data[(i - rowstart) * memacc->c + (j - colstart)];
        }
    }
}
