#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"
#include <assert.h>

/*
With this script we will be timing the performance of different approaches to memcopies between devices and the host.
These tests are to find which approach is the fastest to be used in our matrix multiplication algorithms.
*/

#define GPUTHREADS 32

__global__ void fillMatrix(matrix *A);
void checkMatrixIsOnes(matrix *A);
void checkMatrixIsRows(matrix *A);

int main() {

  int n = 10000, tiledim = n/2, tilesize = tiledim*tiledim*sizeof(double);
  int device, devicecount = 4, NUMTHREADS = devicecount;
  int verbose = 0, rep, reps = 5;
  int i, j;

  //dim3 block(THREADS, THREADS);
  //dim3 grid(ceil(((float)tiledim)/block.x), ceil(((float)tiledim)/block.y));

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop(), results = 0.f;
  printf("Warmup took %f ms\n", elapsedtime);
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
      printf("Using %d opm threads", (int)omp_get_num_threads());
      GPUCHECK(cudaSetDevice(device));
      h_A = kuhdaMallocM1(n, n);
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void)&hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void)&hostbuffer_singlerow[device], tiledim*sizeof(double)));

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
      GPUCHECK(cudaDeviceSynchronize());
  }
  elapsedtime = timer.Stop()
  printf("Allocation took %f ms\n", elapsedtime);


  // 1. Naive approach: pinned buffer on host and cudaMemcpyAsync on one stream, back and forth
  kuhdaFillWithValue(h_A, 1.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = destinations[device][0]; i < destinations[device][1]; ++i){
        for (j = destinations[device][2]; j < destinations[device][3]; ++j){
          // Check this with FRESH MIND
          hostbuffer[device][(i - destinations[device][0]) * tiledim + (j - destinations[device][2])] = h_A->data[i * h_A->c + j];
      }
      GPUCHECK(cudaMemcpyAsync((void*)(&d_A[device]->data[0] + destinations[device][0]*h_A->c + destinations[device][2]), (void*)hostbuffer[device], tilesize, cudaMemcpyHostToDevice, d_streams[device*streamsperdevice]));
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

  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = 0; i < tiledim; ++i){
        for (j = 0; j < tiledim; ++j) d_A->data[i * d_A->c + j] = hostbuffer[device][i * tilesize + j];
      }
      GPUCHECK(cudaMemcpyAsync((void*)hostbuffer[device], (void*)(&h_A->data), tilesize, cudaMemcpyDeviceToHost, d_streams[device*streamsperdevice]));
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

  // 2. TileHostToGPU and TileGPUAddToHost with a single stream
  kuhdaFillWithValue(h_A, 2.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileHostToGPU(0, tiledim, 0, tiledim, h_A, d_A[device], d_streams[device*streamsperdevice]);
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Tiled approach H2D took %f ms\n", results/reps);
  kuhdaFillWithValue(h_A, 0.0);

  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileGPUToHost(0, tiledim, 0, tiledim,d_A[device], h_A, d_streams[device*streamsperdevice]);
    }

    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Tiled approach D2H took %f ms\n", results/reps);

  kuhdaTestForValue(h_A, 1.0, verbose);

  return 0;

  // 3. similar code to TileHostToGPU and TileGPUAddToHost but with special buffer and two streams
  kuhdaFillWithValue(h_A, -5.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = 0; i < tiledim; ++i){
        for (j = 0; j < tiledim; ++j) hostbuffer_singlerow[device][j] = h_A->data[i * h_A->c + j];
        GPUCHECK(cudaMemcpyAsync((void*)(&d_A->data), (void*)hostbuffer_singlerow[device], tiledim*sizeof(double), cudaMemcpyHostToDevice, d_streams[device*streamsperdevice + (int)(i%streamsperdevice)]));
      }
    }
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 0]));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 1]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Fast line H2D took %f ms\n", results/reps);
  kuhdaFillWithValue(h_A, 0.0);

  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = 0; i < tiledim; ++i){
        for (j = 0; j < tiledim; ++j) d_A->data[i * d_A->c + j] = hostbuffer_singlerow[device][j];
        GPUCHECK(cudaMemcpyAsync((void*)hostbuffer_singlerow[device], (void*)(&h_A->data), tiledim*sizeof(double), cudaMemcpyDeviceToHost, d_streams[device*streamsperdevice + (int)(i%streamsperdevice)]));
      }
    }
    #pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 0]));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 1]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Fast line D2H took %f ms\n", results/reps);
  kuhdaTestForValue(h_A, 1.0, verbose);

  // Time the destruction loop
  timer.Start();
  kuhdaFreeM(h_A, 'k');
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      #pragma unroll
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
