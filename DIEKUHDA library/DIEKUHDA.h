
██████╗░ ██╗ ███████╗ ██╗░ ██╗ ██╗░  ██╗ ██╗░ ██╗ ██████╗░  █████╗░     ██╗░ ██╗
██╔══██╗ ██║ ██╔════╝ ██║░██╔╝ ██║░  ██║ ██║░ ██║ ██╔══██╗ ██╔══██╗     ██║░ ██║
██║░ ██║ ██║ █████╗░  █████═╝░ ██║░  ██║ ███████║ ██║░ ██║ ███████║     ███████║
██║░ ██║ ██║ ██╔══╝░  ██╔═██╗░ ██║░  ██║ ██╔══██║ ██║░ ██║ ██╔══██║     ██╔══██║
██████╔╝ ██║ ███████╗ ██║░╚██╗ ╚██████╔╝ ██║░ ██║ ██████╔╝ ██║░ ██║ ██╗ ██║░ ██║
╚═════╝░ ╚═╝ ╚══════╝ ╚═╝░ ╚═╝░ ╚═════╝░ ╚═╝░ ╚═╝ ╚═════╝░ ╚═╝░ ╚═╝ ╚═╝░╚═╝░ ╚═╝
/*
                          /|                        /|
                          | \           __ _ _     / ;
                    ___    \ \   _.-"-" `~"\  `"--' /
                _.-'   ""-._\ ""   ._,"  ; "\"--._./
            _.-'       \./    "-""", )  ~"  |
           / ,- .'          ,     '  `o.  ;  )
           \ ;/       '                 ;   /
            |/        '      |      \   '   |
            /        |             J."\  ,  |
           "         :       \   .'  : | ,. _)
           |         |     /     f |  |`--"--'
            \_        \    \    / _/  |
             \ "-._  _.|   (   j/; -'/
              \  | "/  (   |   /,    |
               | \  |  /\  |\_///   /
               \ /   \ | \  \  /   /
                ||    \ \|  |  |  |
                ||     \ \  |  | /
                |\      |_|/   ||
                L \       ||   ||
                `"'       |\   |\
                          ( \. \ `.
                          |_ _\|_ _\
                            "    "

DTU Special course: Large Scale GPU Computing
  Authors: Simon Aertssen (s181603) and Louis Hein (s181573)
  Supervisors: Bernd Damman and Hans Henrik Brandenborg Soerensen
  DIEKUHDA (pronounce "dcuda"):  Basic data structures, allocation/deallocation
  routines, and input/output routines for matrices, to be used in the

  Version: 1.0 13/03/2020
  Appreciate the ascii art @ fsymbols.com
*/

#ifndef DIEKUHDA_DEFINE
#define DIEKUHDA_DEFINE

/* Macro definitions */
#define DIEKUHDA_FAILURE -1
#define DIEKUHDA_SUCCESS 0
#define DIEKUHDA_MEMORY_ERROR 1
#define DIEKUHDA_FILE_ERROR 2
#define DIEKUHDA_ILLEGAL_INPUT 3
#define DIEKUHDA_DIMENSION_MISMATCH 4
#define MEM_ERR fprintf(stderr,"%s: failed to allocate memory\n",__func__)
#define FAIL_ERR(x) fprintf(stderr,"%s: failure detected, error %d\n",__func__, x)
#define INPUT_NULL_ERR fprintf(stderr,"%s: received NULL pointer as input\n",__func__)
#define INPUT_ILL_ERR_D(x) fprintf(stderr,"%s: received illegal input %d\n",__func__, x)
#define INPUT_ILL_ERR_LF(x) fprintf(stderr,"%s: received illegal input %lf\n",__func__, x)
#define INPUT_ILL_ERR_LU(x) fprintf(stderr,"%s: received illegal input %u\n",__func__, x)


█▀ ▀█▀ █▀█ █░█ █▀▀ ▀█▀ █░█ █▀█ █▀▀ █▀
▄█ ░█░ █▀▄ █▄█ █▄▄ ░█░ █▄█ █▀▄ ██▄ ▄█
/* Structure representing a DIEKUHDA vector */
typedef struct vector {
  unsigned long r;   /* number of elements */
  double * data;     /* pointer to array of length r */
} vector;

/* Structure representing a DIEKUHDA matrix */
typedef struct matrix {
  unsigned long r;   /* number of rows */
  unsigned long c;   /* number of columns */
  double * data;     /* pointer to array of length r*c */
} matrix;

typedef struct euter {
  cublasHandle_t handle;   /* pointer type to an opaque structure holding the cuBLAS library context */
  cudaStream_t *streams;    /* streamIds */
} euter;


█▀▀ █░█ █▄░█ █▀▀ ▀█▀ █ █▀█ █▄░█ █▀
█▀░ █▄█ █░▀█ █▄▄ ░█░ █ █▄█ █░▀█ ▄█
/* Allocation/deallocation on the host*/
vector *kuhdaMallocV(unsigned long r);
void kuhdaFreeV(vector *freethisvector);
matrix *kuhdaMallocM(unsigned long r, unsigned long c);
matrix *kuhdaMallocM1(unsigned long r, unsigned long c);
matrix *kuhdaMallocMdiag(unsigned long r, unsigned long c);
void kuhdaFreeM(matrix *freethismatrix);

/* Allocation/deallocation on the device(s)*/
void kuhdaMatrixToGPU(matrix *hostmatrix);
void kuhdaMatrixToHost(matrix *devicematrix);

/* cuda-specific*/
euter *kuhdaMilchmann(int streamnums);

/* Necessary computations*/
double kuhdaTimeDGEMM(unsigned long m );

#endif
