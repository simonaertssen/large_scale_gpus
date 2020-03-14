#include <stdio.h>
#include <DIEKUHDAlib>
#include </dkd.h>


// This script contains some numerical tests to get to know cublas
// and how to split up matrices in blocks for gpu computation.
// Run with:
// nvcc -lcublas -lgomp experiment_1.cu && ./a.out

int main(){
    printf("Hello world");
}
