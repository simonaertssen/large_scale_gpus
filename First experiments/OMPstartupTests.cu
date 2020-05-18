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

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop();
  printf("Warmup took %f ms\n", elapsedtime);

  // Containers for host and device matrices
  matrix *h_A, *d_A[devicecount];
  double *hostbuffer[devicecount], *hostbuffer_singlerow[devicecount];

  int streamsperdevice = 2;
  int stream, streamcount = streamsperdevice*devicecount;
  cudaStream_t d_streams[streamcount];

  
  // Time the allocation loop
  printf("Timing the (de)allocation loops for n = %d\n", n);
  timer.Start();
  h_A = kuhdaMallocM1(n, n);
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
    GPUCHECK(cudaMallocHost((void**) &hostbuffer[device], tilesize));
    GPUCHECK(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

    for (stream = 0; stream < streamsperdevice; ++stream){
        GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
    }
}
  elapsedtime = timer.Stop();
  printf("Simple allocation took  %f ms\n", elapsedtime);

  timer.Start();
  kuhdaFreeM(h_A, 'k');
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    kuhdaFreeM(d_A[device], 'c');
    cudaFreeHost(hostbuffer[device]);
    cudaFreeHost(hostbuffer_singlerow[device]);

    for (stream = 0; stream < streamsperdevice; ++stream){
      GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
    }
    GPUCHECK(cudaDeviceSynchronize());
}
  elapsedtime = timer.Stop();
  printf("Simple destruction took %f ms\n", elapsedtime);


  // Time the allocation loop
  timer.Start();
  h_A = kuhdaMallocM1(n, n);
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
    GPUCHECK(cudaMallocHost((void**) &hostbuffer[device], tilesize));
    GPUCHECK(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

    #pragma unroll
    for (stream = 0; stream < streamsperdevice; ++stream){
        GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
    }
}
  elapsedtime = timer.Stop();
  printf("Prllel allocation took  %f ms\n", elapsedtime);

  timer.Start();
  kuhdaFreeM(h_A, 'k');
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
    GPUCHECK(cudaSetDevice(device));
    kuhdaFreeM(d_A[device], 'c');
    cudaFreeHost(hostbuffer[device]);
    cudaFreeHost(hostbuffer_singlerow[device]);

    #pragma unroll
    for (stream = 0; stream < streamsperdevice; ++stream){
      GPUCHECK(cudaStreamDestroy(d_streams[stream + streamsperdevice*device]));
    }
    GPUCHECK(cudaDeviceSynchronize());
}
  elapsedtime = timer.Stop();
  printf("Prllel destruction took %f ms\n", elapsedtime);

  return 0;
}
