#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"
#include <assert.h>

/*
With this script we will be timing the performance of different approaches to memcopies between devices and the host.
These tests are to find which approach is the fastest to be used in our matrix multiplication algorithms.
*/

/*
__global__ void fillMatrix(matrix *A);
void checkMatrixIsOnes(matrix *A);
void checkMatrixIsRows(matrix *A);
*/

#define NUMTHREADS 4


int main() {

  int n = 10000, tiledim = n/2, tilesize = tiledim*tiledim*sizeof(double);
  int device, devicecount = 4;
  int verbose = 0, rep, reps = 5;
  int i, j;

  printf("Timing the tiling operations for average of %d reps for matrix size n = %d\n", reps, n);

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop(), results = 0.f;
  printf("Warmup took %f ms\n", elapsedtime);

  // These numbers demarcate the limits of the tiles on the host matrix. For simplicity we are using 4 tiles.
  int destinations[4][4] = {{0, tiledim, 0, tiledim}, {0, tiledim, tiledim, n}, {tiledim, n, 0, tiledim}, {tiledim, n, tiledim, n}};

  // Containers for host and device matrices
	matrix *h_A, *d_A[devicecount];
  double *hostbuffer[devicecount];
  double *hostbuffer_singlerow[devicecount];

  int streamsperdevice = 2;
  int stream, streamcount = streamsperdevice*devicecount;
  cudaStream_t d_streams[streamcount];

  // Time the allocation loop
  timer.Start();
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      h_A = kuhdaMallocM1(n, n);
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void**) &hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
      GPUCHECK(cudaDeviceSynchronize());
  }
  elapsedtime = timer.Stop();
  printf("Allocation took %f ms\n", elapsedtime);

  /* 
  #################################################################################################################
  
  1. Naive approach: pinned buffer of same size as tiles on host and cudaMemcpyAsync on one stream, back and forth

  #################################################################################################################
  */

  results = 0.f;
  kuhdaFillWithValue(h_A, 1.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = 0; i < tiledim; ++i){
        for (j = 0; j < tiledim; ++j){
          hostbuffer[device][i * tiledim + j] = h_A->data[(i + destinations[device][0]) * h_A->c + (j + destinations[device][2])];
        }
      }
      GPUCHECK(cudaMemcpyAsync((void*)(&d_A[device]->data[0]), hostbuffer[device], tilesize, cudaMemcpyHostToDevice, d_streams[device*streamsperdevice]));
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Naive H2D took %f ms\n", results/reps);
  kuhdaFillWithValue(h_A, 0.0);

  results = 0.f;
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaMemcpyAsync(hostbuffer[device], (void*)(&d_A[device]->data[0]), tilesize, cudaMemcpyDeviceToHost, d_streams[device*streamsperdevice]));
      for (i = 0; i < tiledim; ++i){
        for (j = 0; j < tiledim; ++j){
          h_A->data[(i + destinations[device][0]) * h_A->c + (j + destinations[device][2])] = hostbuffer[device][i * tiledim + j];
        }
      }
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Naive D2H took %f ms\n", results/reps);
  kuhdaTestForValue(h_A, 1.0, verbose);

  /* 
  #################################################################################################################
  
  2. TileHostToGPU and TileGPUAddToHost with a single stream. Under the hood pinned memory is used.

  #################################################################################################################
  */
  
  results = 0.f;
  kuhdaFillWithValue(h_A, 2.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileHostToGPU(destinations[device][0], destinations[device][1],destinations[device][2], destinations[device][3], h_A, d_A[device], d_streams[device*streamsperdevice]);
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Tiled H2D took %f ms\n", results/reps);

  kuhdaFillWithValue(h_A, 0.0);

  results = 0.f;
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileGPUToHost(destinations[device][0], destinations[device][1],destinations[device][2], destinations[device][3], d_A[device], h_A, d_streams[device*streamsperdevice]);
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Tiled D2H took %f ms\n", results/reps);

  kuhdaTestForValue(h_A, 2.0, verbose);


  // Time the destruction loop
  timer.Start();
  kuhdaFreeM(h_A, 'k');
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      GPUCHECK(cudaDeviceSynchronize());
      //gpuErrchk(cudaDeviceReset());
  }
  elapsedtime = timer.Stop();
  printf("Destruction took %f ms\n", elapsedtime);

  return 0;
}

/*
__global__ void fillMatrix(matrix *A) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < A->r && col < A->c) A->data[row * A->c + col] = row;
}

void checkMatrixIsOnes(matrix *A) {
  int i, j;
  for (i = 0; i < tilesize; ++i) for (j = 0; j < tilesize; ++j) assert(A->data[i * A->c + j] == 1.0);
}

void checkMatrixIsRows(matrix *A) {
  int i, j;
  for (i = 0; i < tilesize; ++i) for (j = 0; j < tilesize; ++j) assert(A->data[i * A->c + j] == (double)i);
}
*/