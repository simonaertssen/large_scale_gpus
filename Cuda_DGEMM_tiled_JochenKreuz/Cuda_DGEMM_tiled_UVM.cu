// include libraries
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "cublas_v2.h"
#include "cuda.h"

#define nstreams 1    

int main () {

  // banner
  printf ("\n\nGPU DGEMM Exercise\n");
  printf (    "==========================================\n");
  printf (  "\nTiled Matrix-Matrix Multiplication\n");
  printf (    "Using NVIDIA cuBLAS Library\n");

  // echo device data
  int idevice = 0;
  cudaSetDevice(idevice);
  cudaDeviceProp dprops;
  cudaGetDeviceProperties( &dprops, idevice );
  printf ("\nDevice name = %s, with compute capability %d.%d \n", 
	  dprops.name, dprops.major, dprops.minor);

  // define parameters
  int n = 32768;   // matrix dimension - all matrices being multiplied will be square
  int m = 4096;    // tile size - tiles will be square, n must be divisible by m !!
  printf ("\nMatrix sizes: %d x %d, tile size: %d x %d\n", n,n,m,m);
  if ( ( n % m ) != 0  ) {
    printf ("\nmatrix size (n) has to be devisible by tile  size (m) !");
    return 0 ;
  }   
  printf ("Number of Streams: %d", nstreams);
  
  // allocate arrays
  double *a;
  double *b;
  double *c;
  a = (double *) malloc ( n*n*sizeof(double) );
  b = (double *) malloc ( n*n*sizeof(double) );
  c = (double *) malloc ( n*n*sizeof(double) );
  
  // initialize data
  #pragma omp parallel for
  for ( int row = 0; row<n; row++ ) {
    for ( int col = 0; col<n; col++ ) {
      // data in row-major format
      a[row*n+col] = row + col;
      b[row*n+col] = (row == col )  ? 1.0 : 0.0;
      c[row*n+col] = 0.0;
    }
  }

  // create communcations arrays
  double *pa;
  double *pb;
  double *pc;
  cudaMallocManaged ( &pa, m*m*sizeof(double) );
  cudaMallocManaged ( &pb, m*m*sizeof(double) );
  cudaMallocManaged ( &pc, m*m*sizeof(double) );
	  
  // create a handle to cuBlas
  cublasHandle_t cublasHandle;
  cublasCreate( &cublasHandle );

  int ntiles = n/m;
  
  // record start time
  cudaEvent_t t_start;
  cudaEvent_t t_end;
  cudaEvent_t compute_end;
  cudaEventCreate (&t_start);
  cudaEventCreate (&t_end);
  cudaEventCreate (&compute_end);
  cudaEventRecord (t_start,0);

  // caches for indices of previous tile to write back results 
  // from pinned buffer to original result matrix
  int prowtile;
  int pcoltile;

  // PERFORM MULTIPLICATION
  {

    double alpha = 1.0;
    double beta = 1.0; 

    int itile = 0;

    // loop over inner tile dimension
    for ( int iktile = 0; iktile < ntiles; iktile++ ) {
  
      // loop over row tiles
      for ( int irowtile = 0; irowtile < ntiles; irowtile++ ) {

        // loop over column tiles
        for ( int icoltile = 0; icoltile < ntiles; icoltile++ ) {
	  
	  if ( itile >= 1 ) {

	    cudaEventSynchronize (compute_end); // needed since cublasDgemm call is asynchronous and copy is only done 
	                                        // on page fault (if results has already been written to pc
	    // copy result in pinned buffer back to global matrix
            # pragma omp parallel for
	    for ( int i=0; i<m; i++ ) {
	      for ( int j=0; j<m; j++ ) {
		c[(prowtile*m+i)*n+pcoltile*m+j] = pc[i*m+j];
	      }
	    }
	  } 

	  // copy next tile to pinned buffer
          # pragma omp parallel for
	  for ( int i = 0; i < m; i++ ) {
	    for ( int j = 0; j < m; j++ ) {
	      pa[i*m+j] = a[(irowtile*m+i)*n+iktile*m+j];
	      pb[i*m+j] = b[(iktile*m+i)*n+icoltile*m+j];
	      pc[i*m+j] = c[(irowtile*m+i)*n+icoltile*m+j];
	    }
	  }

	  // perform dgemm
	  cublasDgemm ( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, m, m, &alpha, pa, m, pb, m, &beta, pc, m );
	  cudaEventRecord (compute_end, 0);
	  prowtile = irowtile;
	  pcoltile = icoltile;

	  // go to next tile
	  itile++;

	}
      }
    }

    cudaEventSynchronize (compute_end);	  // be sure that last cublasDgemm call has finished
    // copy result in pinned buffer back to source 
    # pragma omp parallel for
    for ( int i=0; i<m; i++ ) {
      for ( int j=0; j<m; j++ ) {
	c[(prowtile*m+i)*n+pcoltile*m+j] = pc[i*m+j];
      }
    }

  } // END OF PERFORM MULTIPLICATION



  // record end time
  cudaEventRecord (t_end,0);
  cudaEventSynchronize(t_end);
  float et;
  cudaEventElapsedTime (&et, t_start, t_end);

  // check results
  printf ("\nchecking results: ");
  bool correct = true;
  int num_errors = 0;
  double abs_error, sum_abs_errors = 0;
# pragma omp parallel for
  for ( int row = 0;  row < n; row++ ) {
    for ( int col = 0; col < n; col++ ) {
      
      abs_error = fabs(c[row * n + col] - a[row * n + col] );
      sum_abs_errors += abs_error;
      if (  abs_error > 10e-1 ) {
	printf ("FAILED\n\nerror: c[%d]: %f != a[%d]: %f", 
		row * n + col,  c[row * n + col], row * n + col,  a[row * n + col]);
	correct = false;
	++num_errors;
	break;
      }
    }
  }
  
  // report results
  if ( correct ) {
    printf ("SUCCESS");
    printf ("\nSum abs errors: %f", sum_abs_errors);
    printf ("\nNumber of errors: %d", num_errors);
    printf("\nExecution time: %4.4f seconds\n", (double)et/1000.);     // cudaEventElapsedTime is in milliseconds
    printf(  "Gflop/s: %4.4f \n\n\n", 2.0e-6*n*n*n/et); // 2( * and + ) *n (inner dimension)*n^2(result size)/(time in ms.)
  } else {
    printf ("\nResult not correct (%d errors), check your code !\n", num_errors);
  }

  // clean up
  cublasDestroy ( cublasHandle );
  cudaEventDestroy ( t_start  );
  cudaEventDestroy ( t_end );

  cudaFree ( pa );
  cudaFree ( pb );
  cudaFree ( pc );

  free (a);
  free (b);
  free (c);

}
