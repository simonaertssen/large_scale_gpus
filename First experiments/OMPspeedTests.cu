#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"
#include <assert.h>

/*
Here we will investigate which OMP approach is the fastest when performing loops in parallel
*/

int main() {

  int n = 10000, tiledim = n/2, tilesize = tiledim*tiledim*sizeof(double);
  int device, devicecount = 4, NUMTHREADS = devicecount;
  int i, j;

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop();
  printf("Warmup took %f ms\n", elapsedtime);

  // Containers for host and device matrices
	matrix *h_A  = kuhdaMallocMP1(n, n), *d_A[devicecount] = NULL;
  double *hostbuffer[devicecount] = NULL, *hostbuffer_singlerow[devicecount] = NULL;

  int streamsperdevice = 2;
  int stream, streamcount = streamsperdevice*devicecount;
  cudaStream_t d_streams[streamcount];

  // Time the allocation loop
  printf("Timing the (de)allocation loops");
  timer.Start();
  for (device = 0; device < devicecount; device++){
      printf("Using %d opm threads", (int)omp_get_num_threads());
      GPUCHECK(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void)&hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void)&hostbuffer_singlerow[device], tiledim*sizeof(double)));

      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }
  elapsedtime = timer.Stop()
  printf("Simple allocation took %f ms\n", elapsedtime);

  timer.Start()
  kuhdaFreeM(h_A, 'p');
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }
  elapsedtime = timer.Stop()
  printf("Simple destruction took %f ms\n", elapsedtime);

  // Time the allocation loop
  timer.Start();
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      printf("Using %d opm threads", (int)omp_get_num_threads());
      GPUCHECK(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void)&hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void)&hostbuffer_singlerow[device], tiledim*sizeof(double)));

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }
  elapsedtime = timer.Stop()
  printf("Parallel allocation took %f ms\n", elapsedtime);

  timer.Start()
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  kuhdaFreeM(h_A, 'p');
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }
  elapsedtime = timer.Stop()
  printf("Parallel destruction took %f ms\n", elapsedtime);


  // Time the tiling operation:
  printf("Timing the tiling operations");
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      printf("Using %d opm threads", (int)omp_get_num_threads());
      GPUCHECK(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void)&hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void)&hostbuffer_singlerow[device], tiledim*sizeof(double)));

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }

  timer.Start()
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    TileHostToGPU(0, tiledim, 0, tiledim, h_A, d_A[device], d_streams[device*streamsperdevice]);
  }
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
  }
  elapsedtime = timer.Stop()
  printf("Simple tile transfer took %f ms\n", elapsedtime);

  timer.Start()
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
  elapsedtime = timer.Stop()
  printf("Parallel tile transfer took %f ms\n", elapsedtime);

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
  timer.Stop();
  printf("Fast parallel tile transfer took %f ms\n", elapsedtime);


  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  kuhdaFreeM(h_A, 'p');
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }

  return 0;
}
