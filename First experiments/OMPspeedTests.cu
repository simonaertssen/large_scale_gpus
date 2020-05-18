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
  int i, j, rep, reps = 5;
  int verbose = 0;
  float results = 0.f;

  Timer timer;
  timer.Start();
  kuhdaWarmup(4);
  float elapsedtime = timer.Stop();
  printf("Warmup took %.1f ms\n", elapsedtime);
  int destinations[4][4] = {{0, tiledim, 0, tiledim}, {0, tiledim, tiledim, n}, {tiledim, n, 0, tiledim}, {tiledim, n, tiledim, n}};

  cudaDeviceProp deviceProp;

  // Containers for host and device matrices
  matrix *h_A, *d_A[devicecount];
  double *hostbuffer[devicecount], *hostbuffer_singlerow[devicecount];

  int streamsperdevice = 2;
  int stream, streamcount = streamsperdevice*devicecount;
  cudaStream_t d_streams[streamcount];

  // Time the tiling operation:
  printf("Timing the tiling operations for average of %d reps for matrix size n = %d\n", reps, n);
  #pragma omp parallel for private(device) num_threads(NUMTHREADS)
  for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));

      // get device properties
      GPUCHECK(cudaGetDeviceProperties(&deviceProp, device));
      // check if support mapped memory
      if (!deviceProp.canMapHostMemory){
          printf("Device %d does not support mapping CPU host memory!\n", device);
          GPUCHECK(cudaDeviceReset());
          exit(EXIT_SUCCESS);
      }
      
      h_A = kuhdaMallocM(n, n);
      d_A[device] = kuhdaMallocDeviceM(tiledim, tiledim);
      GPUCHECK(cudaMallocHost((void**) &hostbuffer[device], tilesize));
      GPUCHECK(cudaMallocHost((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double)));

      //GPUCHECK(cudaHostAlloc((void**) &hostbuffer[device], tilesize, cudaHostAllocMapped));
      //GPUCHECK(cudaHostAlloc((void**) &hostbuffer_singlerow[device], tiledim*sizeof(double), cudaHostAllocMapped));
      //GPUCHECK(cudaHostGetDevicePointer((double**)&(d_A[device]->data), (double*)h_A->data, 0));

      for (stream = 0; stream < streamsperdevice; ++stream){
          GPUCHECK(cudaStreamCreate(&d_streams[stream + streamsperdevice*device]));
      }
  }
  
  results = 0.f;
  kuhdaFillWithValue(h_A, 1.0);
  timer.Start();
  for (rep = 0; rep < reps; rep ++){
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileHostToGPU(destinations[device][0], destinations[device][1],destinations[device][2], destinations[device][3], h_A, d_A[device], d_streams[device*streamsperdevice]);
    }
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Simple tile transfer H2D took %.1f ms\n", results/reps);
  
  results = 0.f;
  kuhdaFillWithValue(h_A, 0.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      TileGPUToHost(destinations[device][0], destinations[device][1],destinations[device][2], destinations[device][3], d_A[device], h_A, d_streams[device*streamsperdevice]);
    }
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice]));
    }
    elapsedtime = timer.Stop();  
    results += elapsedtime; 
  }
  printf("Simple tile transfer D2H took %.1f ms\n", results/reps);
  kuhdaTestForValue(h_A, 1.0, verbose);
  
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
  printf("Parall tile transfer H2D took %.1f ms\n", results/reps);

  results = 0.f;
  kuhdaFillWithValue(h_A, 0.0);
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
  printf("Parall tile transfer D2H took %.1f ms\n", results/reps);
  kuhdaTestForValue(h_A, 2.0, verbose);

  results = 0.f;
  kuhdaFillWithValue(h_A, -5.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    //#pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = destinations[device][0]; i < destinations[device][1]; ++i){
        for (j = destinations[device][2]; j < destinations[device][3]; ++j){
          hostbuffer_singlerow[device][j - destinations[device][2]] = h_A->data[i * h_A->c + j];
        }
        GPUCHECK(cudaMemcpyAsync((void*) (&d_A[device]->data[0] + tiledim * (i - destinations[device][0])), hostbuffer_singlerow[device], tiledim*sizeof(double), cudaMemcpyHostToDevice, d_streams[device*streamsperdevice + (int)(i%streamsperdevice)]));
      }
    }
    //#pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 0]));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 1]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Fast   tile transfer H2D took %.1f ms\n", results/reps);
 
  results = 0.f;
  kuhdaFillWithValue(h_A, 0.0);
  for (rep = 0; rep < reps; rep ++){
    timer.Start();
    //#pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      for (i = destinations[device][0]; i < destinations[device][1]; ++i){
        GPUCHECK(cudaMemcpyAsync(hostbuffer_singlerow[device], (void*) (&d_A[device]->data[0] + tiledim * (i - destinations[device][0])), tiledim*sizeof(double), cudaMemcpyDeviceToHost, d_streams[device*streamsperdevice + (int)(i%streamsperdevice)]));
        for (j = destinations[device][2]; j < destinations[device][3]; ++j){
          h_A->data[i * h_A->c + j] = hostbuffer_singlerow[device][j - destinations[device][2]];
        }
      }
    }

    //#pragma omp parallel for private(device) num_threads(NUMTHREADS)
    for (device = 0; device < devicecount; device++){
      GPUCHECK(cudaSetDevice(device));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 0]));
      GPUCHECK(cudaStreamSynchronize(d_streams[device*streamsperdevice + 1]));
    }
    elapsedtime = timer.Stop();
    results += elapsedtime;
  }
  printf("Fast   tile transfer D2H took %.1f ms\n", results/reps);
  kuhdaTestForValue(h_A, -5.0, verbose);


  printf("Final deallocation\n");
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
      //GPUCHECK(cudaDeviceReset());
  }

  return 0;
}
