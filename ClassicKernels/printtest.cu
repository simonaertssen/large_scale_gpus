#include <stdio.h>

// Exmple doesn't work
__global__ void print_kernel() {
    
    if (threadIdx.x == 1) {
        printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    }
    
}

int main() {
    print_kernel<<<100, 10>>>();
    //cudaDeviceSynchronize();
}