// include libraries
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "cublas_v2.h"
#include "cuda.h"

#define ngpus    4                   // Number of GPUs to use - cant be more than the # of gpus in the node
#define nstreams 3                   // Number of streams PER GPU


// Test for cuda errors
void cuerrmsg ( cudaError_t cuerr, const char *msg) {
  if( cuerr ) {
    printf ("ERROR: %s\n", msg);
    abort();
  }
}

// Test for cuBLAS errors
void custatmsg ( cublasStatus_t custat, const char *msg) {
  if( custat ) {
    printf ("ERROR: %s\n", msg);
    abort();
  }
}


int main () {

  // banner
  printf ("\n\nGPU Computing Advanced Workshop Exercise\n");
  printf (    "==========================================\n");
  printf (  "\nTiled Matrix-Matrix Multiplication\n");
  printf (    "Using cuBlas and streams on multiple GPUs \n");

  // define parameters
  int n = 32768;        // matrix dimension - all matrices being multiplied will be square
  int m = 4096;         // tile size - tiles will be square, n must be divisible by m
                        // on Juron: 40960, 8192

  printf ("\nMatrix sizes: %d x %d, tile size: %d x %d\n", n,n,m,m);
  if ( ( n % m ) != 0  ) {
    printf ("\nmatrix size (n) has to be devisible by tile  size (m) !");
    return 0 ;
  }   
  printf ("Number of GPUs: %d, Number of streams/GPU: %d", ngpus, nstreams);

  // echo device data
  for ( int idevice=0; idevice < ngpus; idevice++ ) {
    cudaSetDevice(idevice);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties( &dprops, idevice );
    printf ("\nDevice name = %s, with compute capability %d.%d \n", 
	    dprops.name, dprops.major, dprops.minor);
  }

  // allocate arrays for A, B and C matrices on the host
  double *a;
  double *b;
  double *c;
  a = (double *) malloc ( n*n*sizeof(double) );
  b = (double *) malloc ( n*n*sizeof(double) );
  c = (double *) malloc ( n*n*sizeof(double) );
  
  // initialize input data on host
#pragma omp parallel for
  for ( int row = 0; row<n; row++ ) {
    for ( int col = 0; col<n; col++ ) {
      // data in row-major format
      a[row*n+col] = row + col;
      b[row*n+col] = (row == col )  ? 1.0 : 0.0;
      c[row*n+col] = 0.0;
    }
  }
  
  
  // create pinned buffers for host<->device communcation
  cudaError_t cuerr;
  double *pa;
  double *pb;
  double *pc;
  cuerr = cudaMallocHost ( &pa, m*m*sizeof(double)*nstreams*ngpus );
  cuerrmsg ( cuerr, "cudaMallocHost pa");
  cuerr = cudaMallocHost ( &pb, m*m*sizeof(double)*nstreams*ngpus );
  cuerrmsg ( cuerr, "cudaMallocHost pb");
  cuerr = cudaMallocHost ( &pc, m*m*sizeof(double)*nstreams*ngpus );
  cuerrmsg ( cuerr, "cudaMallocHost pc");
	  
  // create a handle to cuBlas on each device
  cublasStatus_t custat;
  cublasHandle_t cublasHandle[ngpus]; 
  for ( int igpu=0; igpu<ngpus; igpu++ ) {
    cuerr = cudaSetDevice(igpu);
    if ( cuerr ) {
      printf ("ERROR: cudaSetDevice 1\n");
    }
    custat = cublasCreate( &(cublasHandle[igpu]) );
    if ( custat ) {
      printf ("ERROR createing cuBlas context %d\n", igpu);
    }
    
  }

  // allocate space on device - 3 tiles for a, b, c
  double *d_a[ngpus];
  double *d_b[ngpus];
  double *d_c[ngpus];

  for ( int igpu=0; igpu<ngpus; igpu++ ) {
    cuerr = cudaSetDevice(igpu);
    cuerrmsg ( cuerr, "cudaSetDevice 2");
    cuerr = cudaMalloc ( &(d_a[igpu]), nstreams*m*m*sizeof(double) );
    cuerrmsg ( cuerr, "cudaMalloc d_a[]\n");
    cuerr = cudaMalloc ( &(d_b[igpu]), nstreams*m*m*sizeof(double) );
    cuerrmsg ( cuerr, "cudaMalloc d_b[]\n");
    cuerr = cudaMalloc ( &(d_c[igpu]), nstreams*m*m*sizeof(double) );
    cuerrmsg ( cuerr, "cudaMalloc d_c[]\n");
  }

  int offset = m*m;
  int ntiles = n/m;

  // create streams for each device
  cudaStream_t myStreams[nstreams*ngpus];
  for ( int i=0; i<ngpus*nstreams; i++ ) {
    cudaSetDevice(i/nstreams);
    cuerr = cudaStreamCreate( &myStreams[i] );
    cuerrmsg ( cuerr, "cudaStreamCreate \n");
  }

  // create events to signal when the D2H copy of result tiles has completed
  cudaEvent_t bufferfilled[nstreams*ngpus];
  for ( int i=0; i<ngpus*nstreams; i++ ) {
    cudaSetDevice (i/nstreams);
    cudaEventCreate ( &bufferfilled[i] );
  }

  // record start time
  cudaSetDevice(0);
  cudaEvent_t t_start;
  cudaEvent_t t_end;
  cuerr = cudaEventCreate (&t_start);
  cuerrmsg ( cuerr, "cudaEventCreate t_start \n");
  cuerr = cudaEventCreate (&t_end);
  cuerrmsg ( cuerr, "cudaEventCreate t_end \n");
  cuerr = cudaEventRecord (t_start,0);
  cuerrmsg ( cuerr, "cudaEventRecord \n");

  // caches for indices of previous tiles in streams
  int prowtile[nstreams*ngpus];
  int pcoltile[nstreams*ngpus];

  
  // PERFORM MULTIPLICATION
  {

    double alpha = 1.0;
    double beta = 1.0; 

    int ibuff = 0;
    int itile = 0;
    int igpu = 0;

    // loop over inner tile dimension
    for ( int iktile = 0; iktile < ntiles; iktile++ ) {
  
      // loop over row tiles
      for ( int irowtile = 0; irowtile < ntiles; irowtile++ ) {

        // loop over column tiles
        for ( int icoltile = 0; icoltile < ntiles; icoltile++ ) {

	  cuerr = cudaSetDevice(igpu);
	  cuerrmsg ( cuerr, "cudaSetDevice \n");

	  // first time accessing any device, don't need to empty result buffers 
	  if ( itile >= nstreams*ngpus ) {

	    // make sure that buffers are available
	    cudaEventSynchronize ( bufferfilled[ibuff] );
	    cuerrmsg ( cuerr, "cudaEventSychronize \n" );
	    
	    // copy result in pinned buffer back to source 
            # pragma omp parallel for
	    for ( int i=0; i<m; i++ ) {
	      for ( int j=0; j<m; j++ ) {
		c[(prowtile[ibuff]*m+i)*n+pcoltile[ibuff]*m+j] = pc[ibuff*offset+i*m+j];
	      }
	    }
	  } 

	  // copy data to pinned buffer on host
          # pragma omp parallel for
	  for ( int i=0; i<m; i++ ) {
	    for ( int j=0; j<m; j++ ) {
	      pa[ibuff*offset+i*m+j] = a[(irowtile*m+i)*n+iktile*m+j];
	      pb[ibuff*offset+i*m+j] = b[(iktile*m+i)*n+icoltile*m+j];
	      pc[ibuff*offset+i*m+j] = c[(irowtile*m+i)*n+icoltile*m+j];
	    }
	  }

	  // copy input data to device
	  cuerr = cudaMemcpyAsync ( &(d_a[igpu][(ibuff%nstreams)*offset]), &pa[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
	  cuerrmsg ( cuerr, "cudaMemcpyAsync pa\n");
	  cuerr = cudaMemcpyAsync ( &(d_b[igpu][(ibuff%nstreams)*offset]), &pb[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
	  cuerrmsg ( cuerr, "cudaMemcpyAsync pb\n");
	  cuerr = cudaMemcpyAsync ( &(d_c[igpu][(ibuff%nstreams)*offset]), &pc[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
	  cuerrmsg ( cuerr, "cudaMemcpyAsync pc\n");

	  // perform dgemm
	  custat = cublasSetStream( cublasHandle[igpu], myStreams[ibuff] );
	  custatmsg ( custat, "cublasSetStream \n");
	  custat = cublasDgemm ( cublasHandle[igpu], CUBLAS_OP_T, CUBLAS_OP_T, m, m, m, &alpha, 
				 &(d_a[igpu][(ibuff%nstreams)*offset]), m, 
				 &(d_b[igpu][(ibuff%nstreams)*offset]), m, &beta, 
				 &(d_c[igpu][(ibuff%nstreams)*offset]), m ); 
	  custatmsg ( custat, "cublasDgemm \n");
	  prowtile[ibuff] = irowtile;
	  pcoltile[ibuff] = icoltile;

	  // copy result back to host
	  cuerr = cudaMemcpyAsync ( &pc[ibuff*offset], &(d_c[igpu][(ibuff%nstreams)*offset]), m*m*sizeof(double), cudaMemcpyDeviceToHost, myStreams[ibuff] );
	  cuerrmsg ( cuerr, "cudaMemcpyAsync pc\n");

	  // recored event to signal when D2H copy of result tile is complete
	  cuerr = cudaEventRecord ( bufferfilled[ibuff], myStreams[ibuff] );
	  cuerrmsg ( cuerr, "cudaEventRecord \n" );
	  
	  // update buffer / stream
	  ibuff++;
	  ibuff = ibuff%(nstreams*ngpus);
	  igpu = ibuff/nstreams;
	  itile++;

	}
      }
    }

    ibuff = 0;
    for ( itile=0; itile < nstreams*ngpus; itile ++ ) {

      cuerr = cudaSetDevice ( ibuff/nstreams );
      cuerrmsg ( cuerr, "cudaSetDevice pc\n");

      // make sure that buffers are filled
      cudaEventSynchronize ( bufferfilled[ibuff] );
      cuerrmsg ( cuerr, "cudaEventSychronize \n" );

      // copy result in pinned buffer back to source 
      # pragma omp parallel for
      for ( int i=0; i<m; i++ ) {
	for ( int j=0; j<m; j++ ) {
	  c[(prowtile[ibuff]*m+i)*n+pcoltile[ibuff]*m+j] = pc[ibuff*offset+i*m+j];
	}
      }
	    
      ibuff++;
      ibuff = ibuff%(nstreams*ngpus);

    }

  }

  for ( int i = 0 ; i < ngpus; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
    

  // record end time
  cudaSetDevice(0);
  cudaEventRecord (t_end,0);
  cudaEventSynchronize(t_end);
  float et;
  cudaEventElapsedTime (&et, t_start, t_end);
  
// check results
  printf ("\nchecking results: ");
  bool correct = true;
  double abs_error, sum_abs_errors = 0;
# pragma omp parallel for
  for ( int row = 0;  row < n; row++ ) {
    for ( int col = 0; col < n; col++ ) {
      
      abs_error = fabs(c[row * n + col] - a[row * n + col] );
      sum_abs_errors += abs_error;
      if (  abs_error > 10e-5 ) {
	printf ("FAILED\n\nerror: c[%d]: %f != a[%d]: %f", 
		row * n + col,  c[row * n + col], row * n + col,  a[row * n + col]);
	correct = false;
	break;
      }
    }
  }
  
  // report results
  if ( correct ) {
    printf ("SUCCESS");
    printf ("\nSum abs errors: %f", sum_abs_errors);
    printf("\nExecution time: %4.4f seconds\n", (double)et/1000.);     // cudaEventElapsedTime is in milliseconds
    printf(  "Gflop/s: %4.4f \n\n\n", 2.0e-6*n*n*n/et); // 2( * and + ) *n (inner dimension)*n^2(result size)/(time in ms.)
  } else {
    printf ("\nResult not correct, check your code !\n");
  }

  // clean up
  for ( int igpu=0; igpu<ngpus; igpu++ ) {
    cudaSetDevice ( igpu );
    cudaFree ( d_a[igpu] );
    cudaFree ( d_b[igpu] );
    cudaFree ( d_c[igpu] );
    cublasDestroy ( cublasHandle[igpu] );
  }
  cudaEventDestroy ( t_start  );
  cudaEventDestroy ( t_end );
  cudaFreeHost ( pa );
  cudaFreeHost ( pb );
  cudaFreeHost ( pc );

  free (a);
  free (b);
  free (c);

}
