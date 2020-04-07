#include <stdio.h>
//#include "../DIEKUHDA/kuhda.h"

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

void Execute(double* h_A, double* h_B, int device, int icode){
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
	// matrix *h_A = kuhdaMallocMdiag(n, n);
	// matrix *h_B = kuhdaMallocMdiag(n, n);

	double *h_A = NULL;
	double *h_B = NULL;

	double *d_A[devicecount];
	double *d_B[devicecount];
	double *d_C[devicecount];


	int i;
	for (i = 0; i < devicecount; ++i){
		// gpuErrchk(cudaSetDevice(i));
		// gpuErrchk(cudaMalloc((void**)&d_A[i], tilesize));
		// gpuErrchk(cudaMalloc((void**)&d_B[i], tilesize));
		// gpuErrchk(cudaMalloc((void**)&d_C[i], tilesize));
		printf("Malloc tiles on GPU's.\n");
	}


	// Make the instructions

	// Device 0:
	int instructions0[9] = {icode(GET,A,H), icode(GET,B,H), icode(COMPUTE,C,SELF), icode(GET,A,2), icode(GET,C,1), icode(ADD,C,H),
		icode(COMPUTE,C,SELF), icode(GET,C,1), icode(ADD,C,H)};

	Execute(h_A, h_B, device, icode(SEND,C,SELF));

	return 0;
}
