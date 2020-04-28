#include <stdio.h>
#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c TileMultiplyTest.c && ./a.out

#define print(x) printf("%d\n", x)

// 4 possible COMMANDS:
#define GET 1
#define SEND 2
#define ADD 3
#define COMPUTE 4
// 3 matrices
#define A 0
#define B 1
#define C 2
// get -> source, send -> destination: Device numbers 0, 1, 2, 3
#define H 4 	 // Host = 4
#define SELF 9 // if command is 'compute' no destination or source is required

// transform 3 inputs into a unique 3-digit instruction-code (icode)
#define icode(x, y, z) 100*x + 10*y + z
/*
An INSTRUCTION is an array containing 3 integer values:
1. the command
2. the matrix to be dealt with by the command
3. source / destination or SELF

Examples:
To receive a tile of the A matrix from the host, use
	icode(GET,A,H);  	// translates to icode(1,0,4) and returns the icode '104'

To compute results of the C matrix, use
	icode(COMPUTE,C,SELF); 		// returns icode '429'
*/

// Read the 3-digit instruction-code (icode) and return an instruction array:
int *MakeInstruction(int icode){
	int static instruction[3];
	instruction[0] = icode/100;
	instruction[1] = (icode - instruction[0]*100)/10;
	instruction[2] = icode - (instruction[0]*100) - instruction[1]*10;
	// printf("The instruction-code is: %d -> do %d with %d on %d \n", icode, instruction[0], instruction[1], instruction[2]);
	return instruction;
}

void Execute(matrix* h_A, matrix* h_B, matrix* h_C, matrix* d_A, matrix* d_B, matrix* d_C, int device, int icode){
	// Set the current device:
	// gpuErrchk(cudaSetDevice(device));

	int *instruction = MakeInstruction(icode);

	// Sort on the different commands: GET1 COMPUTE2 ADD3 SEND4
	if (instruction[0] == GET){
		// Sort on devices or host
		if (instruction[2] == H){
			// kuhdaTileToGPU -> make that work with d_A, d_B and d_C
		}

	} else if (instruction[1] == COMPUTE){
		
	} else if (instruction[2] == ADD){

	} else if (instruction[3] == SEND){

	} else {
		printf("An error occured.\n");
	}

}


int main(){
	// Get devicecount:
	int device = 0;
	int devicecount = 4;

	// gpuErrchk(cudaGetDeviceCount(&devicecount));

	unsigned long n = 100, memsize = n*n*sizeof(double);
	unsigned long t = n/2, tilesize = t*t*sizeof(double);
	// Make a giant matrix A and B on the host here:
	// Make these two into a pinned version
	matrix *h_A = kuhdaMallocMdiagP(n, n);
	matrix *h_B = kuhdaMallocMdiagP(n, n);
	matrix *h_C = kuhdaMallocMdiagP(n, n);

	matrix *d_A[devicecount];
	matrix *d_B[devicecount];
	matrix *d_C[devicecount];


	int i;
	for (i = 0; i < devicecount; ++i){
		// gpuErrchk(cudaSetDevice(i));
		d_A[i] = kuhdaMallocDeviceM(t, t);
		d_B[i] = kuhdaMallocDeviceM(t, t);
		d_C[i] = kuhdaMallocDeviceM(t, t);
		
	}
	printf("Malloc tiles on GPU's.\n");


	// Make the instructions

	// Device 0:
	int instructions0[9] = {icode(GET,A,H), icode(GET,B,H), icode(COMPUTE,C,SELF), icode(GET,A,2), icode(GET,C,1), icode(ADD,C,H),
		icode(COMPUTE,C,SELF), icode(GET,C,1), icode(ADD,C,H)};

	// Execute(h_A, h_B, device, icode(SEND,C,SELF));


	// Cleanup
	gpuErrchk(cudaFreeHost(h_A->data));
	gpuErrchk(cudaFreeHost(h_A));
	gpuErrchk(cudaFreeHost(h_B->data));
	gpuErrchk(cudaFreeHost(h_B));
	gpuErrchk(cudaFreeHost(h_C->data));
	gpuErrchk(cudaFreeHost(h_C));

	for (i = 0; i < devicecount; ++i){
		kuhdaFreeM(d_A[i], 'c');
		kuhdaFreeM(d_B[i], 'c');
		kuhdaFreeM(d_C[i], 'c');
	}
	printf("Cleaned up resources.\n");
	return 0;
}
