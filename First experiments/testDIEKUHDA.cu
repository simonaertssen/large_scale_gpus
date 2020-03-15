#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"
#include "cublas_v2.h"

// Run with:
// nvcc ../DIEKUHDA/kuhda.c -lcublas testDIEKUHDA.cu && ./a.out

int main(){
	unsigned long r = 10;
	vector *test = kuhdaMallocV(r);

    printf("Hello world\n");

	return 0;
}
