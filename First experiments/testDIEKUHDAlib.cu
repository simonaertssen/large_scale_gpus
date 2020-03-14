#include <stdio.h>
#include "../DIEKUHDAlib/dkd.h"

// Run with:
// nvcc ../DIEKUHDAlib/dkd.c -lcublas testDIEKUHDAlib.cu && ./a.out

int main(){
	unsigned long r = 10;
	vector *test = kuhdaMallocV(r);

    printf("Hello world\n");

	return 0;
}
