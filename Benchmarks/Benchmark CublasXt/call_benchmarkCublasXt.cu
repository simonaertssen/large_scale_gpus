#include <stdio.h>
#include <stdlib.h>

int main(void){

    int status = 0, rep, reps = 3, iter, maxiter = 2;
    unsigned long n = 0;

    char command[sizeof(unsigned long) + 100];

    for (n = 1024; n <= 65536; n += 1024){
        for (rep = 0; rep < reps; ++rep){
            iter = 0;
            do {
                sprintf(command, "./benchmarkCublasXt %lu", n);
                // printf("command = %s", command);
                status = system(command);
                // printf("status = %d\n", status/255);
                ++iter;
            } while (status/256 != 0 and iter < maxiter);
            if (status/256 != 0 && iter == maxiter) return -1;
        }
    }
    return 0;
}