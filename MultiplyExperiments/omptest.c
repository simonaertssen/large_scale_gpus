#include <stdio.h>
#include "omp.h"

#define NUMTHREADS 4

int main(){ 
    omp_set_num_threads(NUMTHREADS);
    int ID = 0;

    printf("Starting parallel region\n");
    # pragma omp parallel
    {
        ID = omp_get_thread_num();
        printf("ID = %d\n", ID);
    }

    printf("Ending parallel region\n");

    return 0;
}  