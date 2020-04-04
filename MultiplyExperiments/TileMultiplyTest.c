#include <stdio.h>
//#include "../DIEKUHDA/kuhda.h"

// Run with:
// nvcc -lcublas -lgomp ../DIEKUHDA/kuhda.c dataTransfer.c && ./a.out

// These are commands
#define GET 1
#define COMPUTE 2
#define ADD 3
#define SEND 4

#define A 0
#define B 1
#define C 2

#define H 4
#define NEIN 9

#define instr(x, y, z) 100*x + 10*y + z
#define print(x) printf("%d\n", x)
/*
instructions: get comp add send
A B or C = 0, 1, 2
source or destination: get->source, send->destination, host = 4
3 int
*/


int *ReadInstruction(int instruction){
	// Split and read the instruction:
	int static instructions[3];
	instructions[0] = instruction/100;
	instructions[1] = (instruction - instructions[0]*100)/10;
	instructions[2] = instruction - (instructions[0]*100) - instructions[1]*10;
	printf("The instruction is: %d -> do %d with %d on %d \n", instruction, instructions[0], instructions[1], instructions[2]);
	return instructions;
}

void ExecuteCommand(double* h_A, double* h_B, int device, int instruction){
	// Set the current device:
	// gpuErrchk(cudaSetDevice(device));

	int *instructions = ReadInstruction(instruction);

	// Sort on the different commands: GET1 COMPUTE2 ADD3 SEND4
	if (instructions[0] == GET){
		// Sort on devices or host
		if (instructions[2] == H){
			// kuhdaTileToGPU -> make that work with d_A, d_B and d_C
		}

	} else if (instructions[1] == COMPUTE){

	} else if (instructions[2] == ADD){

	} else if (instructions[3] == SEND){

	} else {
		printf("An error occured");
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
		printf("Malloc tiles on GPU's");
	}


	// Make the instructions:
	int instructions0[9] = {instr(GET,A,H), instr(GET,B,H), instr(COMPUTE,C,NEIN), instr(GET,A,2), instr(GET,C,1), instr(ADD,C,H),
		instr(COMPUTE,C,NEIN), instr(GET,C,1), instr(ADD,C,H)};

	ExecuteCommand(h_A, h_B, device, instr(SEND,C,NEIN));

	return 0;
}
