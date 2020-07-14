#include <stdio.h>
#include "../../DIEKUHDA/kuhda.h"
#include "omp.h"

#define NUMTHREADSBUFF 16
#define MAXSTREAMSPERD 2
#define A 0
#define B 1
#define C 2

/*
This script builds on ADM_Direct.cu, but makes use of peer to peer communication between devices.
Each device is associated with different tiles of C, and each device only computes it's own tiles of C.
All jobs are statically scheduled: one for loop over the devices, one for loop for every of the four streams on the device. 
Tiles are sent to the devices and then broadcasted between devices with a fast connection.
For the PS9, the connection between host and devices 2 and 3 is the fastest. Then only half the speed between host and device 0.
So: send to device 0 -> 1 and send to device 2 -> 3

run with
nvcc -o ADM_DirectPeerP9 -O3 -Xcompiler -fopenmp -Xcompiler -mno-float128 -lcublas ../../DIEKUHDA/kuhda.cu ADM_DirectPeerP9.cu && ADM_DirectPeerP9 8192 2048
*/

#define LOG(X,Y) fprintf(logfile, "%s, %s(%d) " #X " " #Y "\n", __TIMESTAMP__, __FILE__, __LINE__);

void TileHostToGPUBuff(	unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *h_matrix, matrix *d_tile, cudaStream_t stream, matrix *memacc );
void TileGPUToHostBuff(unsigned long rowstart, unsigned long rowstop, unsigned long colstart, unsigned long colstop, 
    matrix *d_tile, matrix *h_matrix, cudaStream_t stream, matrix *memacc );


int main(int argc, char* argv[]) {
    // Parallel device warmup by handle creation instead of kuhdaWarmupDevice(device);
    int devicecounter, device, devicecount = 4;
    // int d[4] = {0, 1, 2, 3};
    int d[4] = {0, 3, 1, 2};


    FILE *logfile = fopen("logfile_benchmarkADM_DirectPeerP9.txt", "a");
	// freopen("logfile_benchmarkCublasXt.txt","a",stdout);
	FILE *output = fopen("results_benchmarkADM_DirectPeerP9.txt", "a");
	if (logfile == NULL || output == NULL) {
		fclose(logfile);
		fclose(output);
    return 1;
  	}
	LOG(START, SUCCES);

    omp_set_nested(true);
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
        x = n/2;
    }
    if (argc > 2){
        x = (unsigned long)atoi(argv[2]);
    }    

    // Check dimensions with regards to the available memory: not when measuring block dimension performance
    kuhdaAdjustTileSizeForAvailableMemory(devicecount, n, x);
    if (x > n/2) x = n/2;

    printf("Matrix dimension = %lu, block size = %lu.. \n", n, x);
    int tileop, numtilesperdim = n/x, numtilestotal = numtilesperdim*numtilesperdim, numtilesperdev = numtilestotal/devicecount;
    int streamop, numtilesperstream = numtilesperdev/MAXSTREAMSPERD;
    numtilesperstream = numtilesperstream < 1 ? 1 : numtilesperstream;

    // Containers for host and device matrices
    unsigned long m = n, k = n;    
	matrix *h_A = kuhdaMallocMdiag(n, n); // matrix A as a diagonal matrix
    matrix *h_B = kuhdaMallocM(n, n); // matrix B to be filled with specific values for specific testing
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
    cudaEvent_t streamReady[devicecount][numstreamsperdevice];
    matrix *membuffs[devicecount][numstreamsperdevice][2];

    MatMulTimer timer;
    int firstMemset[devicecount][numstreamsperdevice];

    // Parallel device memory and dependency allocation
    // printf("Allocating tiles A, B and C on %d devices..\n", devicecount);
    #pragma omp parallel for private(device, devicecounter, abc, stream) num_threads(devicecount)
    // Creat all dependencies:
    for (devicecounter = 0; devicecounter < devicecount; ++devicecounter){
        device = d[devicecounter];
        GPUCHECK(cudaSetDevice(device));
    
        #pragma omp parallel for private(stream, abc) num_threads(numstreamsperdevice)
        for (stream = 0; stream < numstreamsperdevice; ++stream){
            for (abc = 0; abc < ABC; ++abc){
                d_All[device][abc][stream] = kuhdaMallocDeviceM(x, x);
            }
            GPUCHECK(cudaStreamCreate(&d_streams[device][stream]));
            GPUCHECK(cudaEventCreateWithFlags(&streamReady[device][stream], cudaEventDisableTiming));
            membuffs[device][stream][0] = kuhdaMallocMP(x, x);
            membuffs[device][stream][1] = kuhdaMallocMP(x, x);
            firstMemset[device][stream] = 1;
        }
    }

    // Main loop counters:
    int Arow, Acol, Brow, Bcol, Crow, Ccol, tileindex, printonce = 1; 

    printf("Computation start..\n");
    timer.Start();

    // Loop over streams per device
    #pragma omp parallel for private(stream, streamop, tileop) num_threads(numstreamsperdevice) 
    for (stream = 0; stream < numstreamsperdevice; ++stream){

        // Loop over all operations on C per stream
        // #pragma omp parallel for private(tileindex, tileop, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(numtilesperstream)
        for (streamop = 0; streamop < numtilesperstream; ++streamop){

            // Loop over all tile operations per stream operation
            for (tileop = 0; tileop < numtilesperdim; ++tileop){
                printf("-------stream %d streamop %d tileop %d \n", stream, streamop, tileop);
                // Parallel device multiplication loop: only here do we actually set the device and perform the transfers
                #pragma omp parallel private(devicecounter, device, tileindex, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(devicecount)
                {
                #pragma omp for
                for (devicecounter = 0; devicecounter < devicecount; devicecounter++){
                    device = d[devicecounter];
                    GPUCHECK(cudaSetDevice(device));
                    // Register indices of C tiles
                    tileindex = devicecounter + (stream*numtilesperstream + streamop)*devicecount;
                    Crow = tileindex/numtilesperdim; Ccol = tileindex%numtilesperdim;

                    // Set contents of C to zero for use as an accumulator:
                    if (tileop == 0 && firstMemset[device][stream] == 0) {
                        GPUCHECK(cudaMemsetAsync(d_All[device][C][stream]->data, 0, x*x*sizeof(double), d_streams[device][stream]));
                    } else {
                        firstMemset[device][stream] = 0;
                    }

                    // Loop over all tiles of A and B to copy: Arow = Crow and Bcol = Ccol
                    Arow = Crow;   Acol = tileop;
                    Brow = tileop; Bcol = Ccol;

                    // Copy the A-tile between devices, using the same stream number of the devices.
                    // If the following C tile still has the same row of A, send the tile between devices, otherwise not.
                    // Check first if the current tile is not the last in the row (+1 for 0-based indexing).
                    // Then check whether the next tile is on the same row, which should be the same multiple of numtilesperdim.
                    if (devicecounter == 0 || devicecounter == 2){
                        TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][A][stream], d_streams[device][stream], membuffs[device][stream][0]); // Tile A
                        GPUCHECK(cudaStreamSynchronize(d_streams[device][stream]));
                        // Now send from 0->2 and from 1->3
                        if (((tileindex+1) % numtilesperdim != 0) && Crow == (tileindex+1)/numtilesperdim){
                            GPUCHECK(cudaMemcpyPeerAsync(d_All[d[devicecounter+1]][A][stream]->data, d[devicecounter], d_All[device][A][stream]->data, device, x*x*sizeof(double), d_streams[device][stream]));
                            if (printonce) printf("%d: Sending from %d to %d\n", device, tileindex, tileindex+1);
                            GPUCHECK(cudaEventRecord(streamReady[device][stream], d_streams[device][stream]));
                        } else {
                            if (printonce) printf("%d: No sending possible from %d to %d\n", device, tileindex, tileindex+1);
                        }
                    }
                }
                
                // Now get the tiles if transferred between devices:
                //#pragma omp parallel for private(device, tileindex, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(devicecount)
                #pragma omp for
                for (devicecounter = 0; devicecounter < devicecount; devicecounter++){
                    device = d[devicecounter];
                    GPUCHECK(cudaSetDevice(device));
                    tileindex = devicecounter + (stream*numtilesperstream + streamop)*devicecount;
                    Crow = tileindex/numtilesperdim; Ccol = tileindex%numtilesperdim; Arow = Crow; Acol = tileop; Brow = tileop; Bcol = Ccol;

                    if (devicecounter == 1 || devicecounter == 3){
                        if (Crow == (tileindex-1)/numtilesperdim){
                            GPUCHECK(cudaStreamWaitEvent(d_streams[device][stream], streamReady[d[devicecounter-1]][stream], 0));
                            if (printonce) printf("%d: Receiving from %d on %d\n", device, tileindex-1, tileindex);
                        } else {
                            if (printonce) printf("%d: No receiving possible from %d on %d\n", device, tileindex-1, tileindex);
                            TileHostToGPUBuff(Arow*x, (Arow+1)*x, Acol*x, (Acol+1)*x, h_A, d_All[device][A][stream], d_streams[device][stream], membuffs[device][stream][0]); // Tile A
                        }
                    }
                    // Now that the tile of A is transferred, get B and start computation
                    TileHostToGPUBuff(Brow*x, (Brow+1)*x, Bcol*x, (Bcol+1)*x, h_B, d_All[device][B][stream], d_streams[device][stream], membuffs[device][stream][1]); // Tile B
                    
                    kuhdammson(d_All[device][A][stream], d_All[device][B][stream], d_All[device][C][stream], d_streams[device][stream], handles[device]);
                    GPUCHECK(cudaStreamSynchronize(d_streams[device][stream]));
                }
            }
            }

            #pragma omp parallel for private(device, tileindex, Arow, Acol, Brow, Bcol, Crow, Ccol) num_threads(devicecount)
            for (devicecounter = 0; devicecounter < devicecount; devicecounter++){
                device = d[devicecounter];
                GPUCHECK(cudaSetDevice(device));
                tileindex = devicecounter + (stream*numtilesperstream + streamop)*devicecount;
                Crow = tileindex/numtilesperdim; Ccol = tileindex%numtilesperdim; Arow = Crow; Acol = tileop; Brow = tileop; Bcol = Ccol;

                TileGPUToHostBuff(Crow*x, (Crow+1)*x, Ccol*x, (Ccol+1)*x, d_All[device][C][stream], h_C, d_streams[device][stream], membuffs[device][stream][0]);
                printf("sending tile %d back\n", tileindex);
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
    fprintf(output, "%zu, %d, %.1lf\n", n, x, timingResult);

    // Free all dependencies
    // printf("Cleaning up..\n");
    GPUCHECK(cudaSetDevice(0));

	kuhdaFreeM(h_A, 'k');
	kuhdaFreeM(h_B, 'k');
    kuhdaFreeM(h_C, 'k');

    timer.Release();
 
    #pragma omp parallel for private(device, devicecounter, abc, stream) num_threads(devicecount) 
    for (devicecounter = 0; devicecounter < devicecount; ++devicecounter){
        device = d[devicecounter];
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
            GPUCHECK(cudaEventDestroy(streamReady[device][stream]));
        }
        // Takes NO arguments
        GPUCHECK(cudaDeviceReset());
    }

    LOG(STOP, SUCCES);
	fclose(logfile);
	fclose(output);
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
