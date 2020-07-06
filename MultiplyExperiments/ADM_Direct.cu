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
void TileGPUToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
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
    unsigned long n, x;
    if (argc > 1){
        n = (unsigned long)atoi(argv[1]);
        // if (n >= 73029) {
        //     n = 73028;
        //     // In this case, n exceeds the size necessary for 128 GB of data on the host for three matrices: sqrt(128 * pow(1000, 3) / 8 / 3) = 73029
        //     printf("Matrix dimension too large, setting matrix size to %d..\n", n);
        // }
    }
    if (argc > 2){
        x = (unsigned long)atoi(argv[2]);
        // x = n/numtilesperdim;
        // if (x%n != 0) x = n/4;
        // if (x > n ) {
        //     x = n/2;
        //     printf("Block size too large, setting block size to %d..\n", x);
        // } else if (x >= 63245){
        //     // In this case, x exceeds the size necessary for 32 GB of data on a device: sqrt(32 * pow(1000, 3) / 8) = 63245
        //     x /= 2;
        // } else if (numtilesperdim >= 512){
        //     // Then we register the number of tiles
        //     x = numtilesperdim;
        //     numtilesperdim = n/x;
        // }
    }    

    // Check dimensions with regards to the available memory:
    kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);

    printf("Matrix dimension = %lu, block size = %lu.. \n", n, x);
    int tileop, numtilesperdim = n/x, numtilestotal = numtilesperdim*numtilesperdim, numtilesperdev = numtilestotal/devicecount;
    int streamop, numtilesperstream = numtilesperdev/MAXSTREAMSPERD;
    numtilesperstream = numtilesperstream < 1 ? 1 : numtilesperstream;

    // Containers for host and device matrices
    unsigned long m = n, k = n;    
	matrix *h_A = kuhdaMallocMdiag(n, n); // matrix A as a diagonal matrix
    matrix *h_B = kuhdaMallocMdiag(n, n); // matrix B to be filled with specific values for specific testing
    matrix *h_C = kuhdaMallocM(n, n);     // matrix C will contain results: same values at each spot as in b
    unsigned long i, j;
    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF)
	for (i = 0; i < h_B->r; ++i){
		for (j = 0; j < h_B->c; ++j){
            h_B->data[i*h_B->c + j] = (i + j) * 0.1 + i;
        }
    }

    // Counters for streams: number of streams is number of operations per device, but adjust for less streams if large tiles
    int stream, numstreamsperdevice = numtilesperdev > MAXSTREAMSPERD ? MAXSTREAMSPERD : numtilesperdev;
    printf("numtilestotal = %d, numtilesperdev = %d, numtilesperstream = %d, numstreamsperdevice = %d\n", numtilestotal, numtilesperdev, numtilesperstream, numstreamsperdevice);
    
    int abc, ABC = 3; 
    matrix *d_All[devicecount][ABC][numstreamsperdevice];   // matrix tiles on each device per stream

    // Cuda dependencies
    cudaStream_t d_streams[devicecount][numstreamsperdevice];
    matrix *membuffs[devicecount][numstreamsperdevice][2];

    MatMulTimer timer;

    // Parallel device memory and dependency allocation
    printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, abc, stream) num_threads(devicecount)
    // Creat all dependencies:
    for (device = 0; device < devicecount; ++device){
        GPUCHECK(cudaSetDevice(device));

        #pragma omp parallel for private(stream, abc) num_threads(numstreamsperdevice)
        for (stream = 0; stream < numstreamsperdevice; ++stream){
            for (abc = 0; abc < ABC; ++abc){
                d_All[device][abc][stream] = kuhdaMallocDeviceM(x, x);
            }
            GPUCHECK(cudaStreamCreate(&d_streams[device][stream]));
            // GPUCHECK(cudaStreamCreateWithFlags(&d_streams[device][stream], cudaStreamNonBlocking));
            membuffs[device][stream][0] = kuhdaMallocMP(x, x);
            membuffs[device][stream][1] = kuhdaMallocMP(x, x);
        }
    }

    // Main loop counters:
    int Arow, Acol, Brow, Bcol, Crow, Ccol, tileindex;

    printf("Computation start..\n");
    timer.Start();

    // Parallel device multiplication loop
    #pragma omp parallel for num_threads(devicecount)
    for (device = 0; device < devicecount; device++){
        GPUCHECK(cudaSetDevice(device));

        // Loop over streams per device
        // #pragma omp parallel for num_threads(numstreamsperdevice)
        #pragma omp parallel for private(stream, streamop, tileindex, tileop, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(numstreamsperdevice) 
        for (stream = 0; stream < numstreamsperdevice; ++stream){
            // Loop over all operations on C per stream
            // #pragma omp parallel for private(tileindex, tileop, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(numtilesperstream)
            for (streamop = 0; streamop < numtilesperstream; ++streamop){
                // Register indices of C tiles
                // tileindex = (device*numstreamsperdevice + stream)*numtilesperstream + streamop; 
                tileindex = (stream*devicecount + device)*numtilesperstream + streamop; 
                Crow = tileindex/numtilesperdim; Ccol = tileindex%numtilesperdim;

                // Set contents of C to zero for use as an accumulator:
                GPUCHECK(cudaMemsetAsync(d_All[device][C][stream]->data, 0, x*x*sizeof(double), d_streams[device][stream]));
                // printf("Dev %d, stream %d: tileindex = %d, (%d,%d)\n", device, stream, tileindex, Crow, Ccol);

                // Loop over all tiles of A and B to copy: Arow = Crow and Bcol = Ccol
                // #pragma omp parallel for private(tileop, Arow, Acol, Brow, Bcol) num_threads(numtilesperdim)
                for (tileop = 0; tileop < numtilesperdim; ++tileop){
                    Arow = Crow;   Acol = tileop;
                    Brow = tileop; Bcol = Ccol;

                    TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][A][stream], d_streams[device][stream], membuffs[device][stream][0]); // Tile A
                    TileHostToGPUBuff(Brow*x, (Brow+1)*x, Bcol*x, (Bcol+1)*x, h_B, d_All[device][B][stream], d_streams[device][stream], membuffs[device][stream][1]); // Tile B
                    
                    GPUCHECK(cudaStreamSynchronize(d_streams[device][stream]));
                    kuhdammson(d_All[device][A][stream], d_All[device][B][stream], d_All[device][C][stream], d_streams[device][stream], handles[device]);
                }

                TileGPUToHostBuff(Crow*x, (Crow+1)*x, Ccol*x, (Ccol+1)*x, d_All[device][C][stream], h_C, d_streams[device][stream], membuffs[device][stream][0]);
            }
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

        #pragma omp parallel for private(stream, abc) num_threads(numstreamsperdevice)
        for (stream = 0; stream < numstreamsperdevice; ++stream){
            for (abc = 0; abc < ABC; ++abc){
                kuhdaFreeM(d_All[device][abc][stream], 'c');
            }
            GPUCHECK(cudaStreamDestroy(d_streams[device][stream]));
            kuhdaFreeM(membuffs[device][stream][0], 'p');
            kuhdaFreeM(membuffs[device][stream][1], 'p');
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

    unsigned long i, j; //cols = colstop - colstart, rows = rowstop - rowstart, i, j;

    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF) collapse(2)
    for (i=rowstart; i<rowstop; ++i){
        for (j=colstart; j<colstop; ++j){
            memacc->data[(i - rowstart) * memacc->c + (j - colstart)] = h_matrix->data[i * h_matrix->c + j];
        }
    }
    
    // GPUCHECK(cudaMemcpyAsync((void*)&d_tile->data[0], (void*)&memacc->data[0], rows*cols*sizeof(double), cudaMemcpyHostToDevice, stream));
    GPUCHECK(cudaMemcpy2DAsync((void*)&d_tile->data[0], memacc->c*sizeof(double), (const void*)&memacc->data[0], memacc->c*sizeof(double), memacc->c*sizeof(double), memacc->r, cudaMemcpyHostToDevice, stream));

}

void TileGPUToHostBuff( unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc ){
    if (h_matrix == NULL || d_tile == NULL) INPUT_NULL_ERR;
    if (rowstart > rowstop) INPUT_ILL_ERR_LU(rowstop);
    if (colstart > colstop)	INPUT_ILL_ERR_LU(colstop);
    if (h_matrix->r <= 0 || h_matrix->c <= 0 || d_tile->r <= 0 || d_tile->c <= 0) INPUT_ILL_ERR_LU(h_matrix->r);
    if (stream == NULL) INPUT_NULL_ERR;

    unsigned long i, j; //cols = colstop - colstart, rows = rowstop - rowstart, i, j;
    // GPUCHECK(cudaMemcpyAsync((void*)&memacc->data[0], (void*)&d_tile->data[0], rows*cols*sizeof(double), cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaMemcpy2DAsync((void*)&memacc->data[0], memacc->c*sizeof(double), (const void*)&d_tile->data[0], d_tile->c*sizeof(double), d_tile->c*sizeof(double), d_tile->r, cudaMemcpyDeviceToHost, stream));
    GPUCHECK(cudaStreamSynchronize(stream));

    #pragma omp parallel for private(i,j) num_threads(NUMTHREADSBUFF) collapse(2)
    for (i = rowstart; i < rowstop; ++i){
        for (j = colstart; j < colstop; ++j){
            h_matrix->data[i * h_matrix->c + j] = memacc->data[(i - rowstart) * memacc->c + (j - colstart)];
        }
    }
}
