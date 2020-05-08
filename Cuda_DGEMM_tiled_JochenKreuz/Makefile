NVCC = nvcc
OPT_FLAGS = -O3
OMP_FLAGS = -Xcompiler -fopenmp
CUBLASFLAGS = -lcublas

#JURECA (K80)
#COMPUTE_CAPABILITY=37
#JURON (P100)
#COMPUTE_CAPABILITY=60
#JUWELS (V100)
COMPUTE_CAPABILITY=70

NVFLAGS = --cudart=static -gencode arch=compute_$(COMPUTE_CAPABILITY),code=compute_$(COMPUTE_CAPABILITY) -std=c++11


.phony: all
all: Cuda_DGEMM_tiled Cuda_DGEMM_tiled_streams Cuda_DGEMM_tiled_streams_multigpu Cuda_DGEMM_tiled_UVM Cuda_DGEMM_tiled_streams_UVM Cuda_DGEMM_cublasXt

Cuda_DGEMM_tiled: Cuda_DGEMM_tiled.cu
	$(NVCC) $(OPT_FLAGS) $(NVFLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled Cuda_DGEMM_tiled.cu 

Cuda_DGEMM_tiled_streams: Cuda_DGEMM_tiled_streams.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams Cuda_DGEMM_tiled_streams.cu 

Cuda_DGEMM_tiled_streams_multigpu: Cuda_DGEMM_tiled_streams_multigpu.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams_multigpu Cuda_DGEMM_tiled_streams_multigpu.cu

Cuda_DGEMM_tiled_UVM: Cuda_DGEMM_tiled_UVM.cu
	$(NVCC) $(OPT_FLAGS) $(NVFLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_UVM Cuda_DGEMM_tiled_UVM.cu 

Cuda_DGEMM_tiled_streams_UVM: Cuda_DGEMM_tiled_streams_UVM.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams_UVM Cuda_DGEMM_tiled_streams_UVM.cu 

Cuda_DGEMM_cublasXt: Cuda_DGEMM_cublasXt.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_cublasXt Cuda_DGEMM_cublasXt.cu

.phony: clean
clean:
	rm -f Cuda_DGEMM_tiled
	rm -f Cuda_DGEMM_tiled_streams 
	rm -f Cuda_DGEMM_tiled_streams_multigpu
	rm -f Cuda_DGEMM_tiled_UVM
	rm -f Cuda_DGEMM_tiled_streams_UVM
	rm -f Cuda_DGEMM_cublasXt

.phony: run
run: run_task1 run_task2 run_task3 run_task1_UVM run_task2_UVM run_cublasXt

.phony: run_task1
run_task1: Cuda_DGEMM_tiled
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_tiled

.phony: run_task2
run_task2: Cuda_DGEMM_tiled_streams
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_tiled_streams

.phony: run_task3
run_task3: Cuda_DGEMM_tiled_streams_multigpu
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_tiled_streams_multigpu

.phony: run_task1_UVM
run_task1_UVM: Cuda_DGEMM_tiled_UVM
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_tiled_UVM

.phony: run_task2_UVM
run_task2_UVM: Cuda_DGEMM_tiled_streams_UVM
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_tiled_streams_UVM

.phony: run_cublasXt
run_cublasXt: Cuda_DGEMM_cublasXt
	${CUDA19_SUBMIT_CMD} ./Cuda_DGEMM_cublasXt 32768 4096 4


# to include data transfers in timeline and have counters availble for kernel analysis, two runs are required
# both files can be loaded into NVVP  ("Timeline datafile", "Event/Metric data files")
.phony: profile_task1
profile_task1: Cuda_DGEMM_tiled
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_tiled.out -f ./Cuda_DGEMM_tiled
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_tiled.metric.out -f ./Cuda_DGEMM_tiled

.phony: profile_task2
profile_task2: Cuda_DGEMM_tiled_streams
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_tiled_streams.out -f ./Cuda_DGEMM_tiled_streams
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_tiled_streams.metric.out -f ./Cuda_DGEMM_tiled_streams

.phony: profile_task3
profile_task3: Cuda_DGEMM_tiled_streams_multigpu
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_tiled_streams_multigpu.out -f ./Cuda_DGEMM_tiled_streams_multigpu
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_tiled_streams_multigpu.metric.out -f ./Cuda_DGEMM_tiled_streams_multigpu

.phony: profile_task1_UVM
profile_task1_UVM: Cuda_DGEMM_tiled_UVM
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_tiled_UVM.out -f ./Cuda_DGEMM_tiled_UVM
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_tiled_UVM.metric.out -f ./Cuda_DGEMM_tiled_UVM

.phony: profile_task2_UVM
profile_task2_UVM: Cuda_DGEMM_tiled_streams_UVM
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_tiled_streams_UVM.out -f ./Cuda_DGEMM_tiled_streams_UVM
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_tiled_streams_UVM.metric.out -f ./Cuda_DGEMM_tiled_streams_UVM

.phony: profile_cublasXt
profile_cublasXt: Cuda_DGEMM_cublasXt
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device -o nvprof_Cuda_DGEMM_cublasXt.out -f ./Cuda_DGEMM_cublasXt
	${CUDA19_SUBMIT_CMD} nvprof --unified-memory-profiling per-process-device --analysis-metrics -o nvprof_Cuda_DGEMM_cublasXt.metric.out -f ./Cuda_DGEMM_cublasXt
