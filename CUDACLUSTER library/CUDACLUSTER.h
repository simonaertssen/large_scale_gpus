/* DTU Special course: Large Scale GPU Computing
    
    Authors: 
    Simon Aertssen - s181603
    Louis Hein     - s181573

    Supervisors:
    Bernd Damman
    Hans Henrik Brandenborg Soerensen

	CUDACLUSTER (CUDACLUS):  Basic data structures, allocation/deallocation
	routines, and input/output routines for matrices.

  Version: 1.0 13/03/2020
*/


#ifndef CUDACLUSTER_DEFINE
#define CUDACLUSTER_DEFINE

#include <stdio.h>

/* Macro definitions */
#define CUDACLUSTER_FAILURE -1
#define CUDACLUSTER_SUCCESS 0
#define CUDACLUSTER_MEMORY_ERROR 1
#define CUDACLUSTER_FILE_ERROR 2
#define CUDACLUSTER_ILLEGAL_INPUT 3
#define CUDACLUSTER_DIMENSION_MISMATCH 4
#define MEM_ERR fprintf(stderr,"%s: failed to allocate memory\n",__func__)
#define FILE_ERR(x) fprintf(stderr,"%s: failed to open file %s\n",__func__,x)
#define INPUT_ERR fprintf(stderr,"%s: received NULL pointer as input\n",__func__)


/* Structure representing a CUDACLUSTER vector */
typedef struct vector {
  unsigned long r;   /* number of elements */
  double * data;     /* pointer to array of length r */
} vector;

/* Structure representing a CUDACLUSTER matrix */
typedef struct matrix {
  unsigned long r;   /* number of rows */
  unsigned long c;   /* number of columns */
  double * data;     /* pointer to array of length r*c */
} matrix;


/**********************/
/* Prototypes         */
/**********************/

/* Allocation/deallocation */
/* vector *malloc_vector(unsigned long r);
void free_vector(vector *freethisvector); */
matrix *malloc_matrix(unsigned long r, unsigned long c);
matrix *malloc_zeros(unsigned long r, unsigned long c);
matrix *malloc_ones(unsigned long r, unsigned long c);
void free_matrix(matrix *freethismatrix);

#endif
