#!/bin/bash
### General options
### specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J optimalBlockDimCublasXt
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:45
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]" 
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
#BSUB -R "span[hosts=1]"
#BSUB -u s181603@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o %J.out
#BSUB -e %J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2

N="1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 26624 27648 28672 29696 30720 31744 32768"

for n in $N
do

    ./benchmarkCublasXt $n

done


