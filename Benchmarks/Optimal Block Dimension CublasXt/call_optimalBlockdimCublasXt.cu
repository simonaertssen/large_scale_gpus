#include <stdio.h>
#include <stdlib.h>

int main(void){

    int status = 0, rep, reps = 3, iter, maxiter = 5;
    unsigned long n = 32768, blockdim = 0;

    char command[2*sizeof(unsigned long) + 100];

    for (blockdim = 1024; blockdim <= 1638; blockdim *= 2){
        for (rep = 0; rep < reps; ++rep){
            iter = 0;
            do {
                sprintf(command, "./optimalBlockdimCublasXt %lu %lu", n, blockdim);
                // sprintf(command, "./Optimal\\ Block\\ Dimension\\ CublasXt/optimalBlockdimCublasXt %lu %lu", n, blockdim);

                // printf("command = %s", command);
                status = system(command);
                // printf("status = %d\n", status/255);
                ++iter;
            } while (status/256 != 0 and iter < maxiter);
        }
    }
    return 0;
}