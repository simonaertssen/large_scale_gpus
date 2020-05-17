#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "omp.h"
#include <assert.h>

// run with 
// nvcc -O3 -Xcompiler -fopenmp -lcublas ../DIEKUHDA/kuhda.c OMPspeedTest.c && ./a.out

/*
Here we will investigate which OMP approach is the fastest when performing loops in parallel
*/

#define NUMTHREADS 4


int main() {

  int n = 10000, tiledim = n/2, tilesize = tiledim*tiledim*sizeof(double);
  int device, devicecount = 4;
  int i, j;

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop();
  printf("Warmup took %f ms\n", elapsedtime);

  // Containers for host and device matrices
  matrix *h_A  = kuhdaMallocMP1(n, n), *d_A[devicecount];
  //double *test = (double*) malloc(tilesize);
  double *hostbuffer[devicecount], *hostbuffer_singlerow[devicecount];

  int streamsperdevice = 2;
  int stream, streamcount = streamsperdevice*devicecount;
  cudaStream_t d_streams[streamcount];

  

  /*
  // Time the allocation loop
  printf("Timing the (de)allocation loops\n");
  timer.Start();
  for (device = 0; device < devicecount; device++){
      printf("Using %d OMP threads\n", (int)omp_get_num_threads());
      gpuErrchk(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      gpuErrchk(cudaMallocHost((void**) &hostbuffer[device], tilesize));
      gpuErrchk(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

      for (stream = 0; stream < streamsperdevice; ++stream){
          gpuErrchk(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }
  elapsedtime = timer.Stop();
  printf("Simple allocation took %f ms\n", elapsedtime);

  timer.Start();
  kuhdaFreeM(h_A, 'p');
  for (device = 0; device < devicecount; device++){
      gpuErrchk(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      for (stream = 0; stream < streamsperdevice; ++stream){
          gpuErrchk(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }
  elapsedtime = timer.Stop();
  printf("Simple destruction took %f ms\n", elapsedtime);

  
  // Time the allocation loop
  timer.Start();
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      printf("Using %d OMP threads", (int)omp_get_num_threads());
      gpuErrchk(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      gpuErrchk(cudaMallocHost((void**) &hostbuffer[device], tilesize));
      gpuErrchk(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          gpuErrchk(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }
  elapsedtime = timer.Stop();
  printf("Parallel allocation took %f ms\n", elapsedtime);

  timer.Start();
  kuhdaFreeM(h_A, 'p');
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      gpuErrchk(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFree(hostbuffer[device]);
      cudaFree(hostbuffer_singlerow[device]);

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          gpuErrchk(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }
  elapsedtime = timer.Stop();
  printf("Parallel destruction took %f ms\n", elapsedtime);

 */

  // Time the tiling operation:
  printf("Timing the tiling operations\n");
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      printf("Using %d OMP threads\n", (int)omp_get_num_threads());
      gpuErrchk(cudaSetDevice(device));
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      gpuErrchk(cudaMallocHost((void**) &hostbuffer[device], tilesize));
      gpuErrchk(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
          gpuErrchk(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }

  timer.Start();
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    TileHostToGPU(0, tiledim, 0, tiledim, h_A, d_A[device], d_streams[device*streamsperdevice]);
  }
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    gpuErrchk(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
  }
  elapsedtime = timer.Stop();
  printf("Simple tile transfer took %f ms\n", elapsedtime);

  timer.Start();
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    TileHostToGPU(0, tiledim, 0, tiledim, h_A, d_A[device], d_streams[device*streamsperdevice]);
  }
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    gpuErrchk(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
  }
  elapsedtime = timer.Stop();
  printf("Parallel tile transfer took %f ms\n", elapsedtime);

  timer.Start();
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    for (i = 0; i < tiledim; ++i){
      for (j = 0; j < tiledim; ++j) hostbuffer_singlerow[device][j] = h_A->data[i * h_A->c + j];
      printf("number = %d\n", i%streamsperdevice);
      d_streams[device*streamsperdevice + (int)(i%streamsperdevice)];
      //gpuErrchk(cudaMemcpyAsync((void*)(&d_A[device]->data[0] + tiledim * i), (void*)hostbuffer_singlerow[device], tiledim*sizeof(double), 
      //                                  cudaMemcpyHostToDevice, d_streams[device*streamsperdevice + (int)(i%streamsperdevice)]));
    }
  }

  
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    gpuErrchk(cudaSetDevice(device));
    gpuErrchk(cudaStreamSynchronize(d_streams[device*streamsperdevice + 0]));
    gpuErrchk(cudaStreamSynchronize(d_streams[device*streamsperdevice + 1]));
  }
  elapsedtime = timer.Stop();
  printf("Fast parallel tile transfer took %f ms\n", elapsedtime);

  kuhdaFreeM(h_A, 'p');
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      gpuErrchk(cudaSetDevice(device));
      kuhdaFreeM(d_A[device], 'c');
      cudaFreeHost(hostbuffer[device]);
      cudaFreeHost(hostbuffer_singlerow[device]);

      #pragma unroll
      for (stream = 0; stream < streamsperdevice; ++stream){
        d_streams[stream + streamsperdevice*device];
        gpuErrchk(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
      }
      gpuErrchk(cudaDeviceReset());
  }

  return 0;
}
