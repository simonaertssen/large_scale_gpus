#!/bin/bash
### General options
### specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J optimalBlockDimcublasXt
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]" 
#BSUB -R "select[sxm2]"
#BSUB -R "span[hosts=1]"
#BSUB -u s181603@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -J blockdimTest
#BSUB -o blockdimTest_%J.out
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2

N="5000 10000"
BD="1024 2048 2500 4096 5000"

for n in $N
do
    for b in $BD
        
    do
        ./optimalBlockDimcublasXt $n $b
    done

done


