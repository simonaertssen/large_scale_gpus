TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= matmult_lib.o matmult_gpu1.o matmult_gpu2.o matmult_gpu3.o matmult_gpu4.o matmult_gpu5.o matmult_gpu6.o matmult_gpulib.o

OPT	= -g -O3
PIC = -fpic
OMP   = -fopenmp
XPIC  = -Xcompiler -fpic --default-stream per-thread
XOPT  = -Xptxas=-v -lineinfo #-G # only use -G for debugging / profiler
XARCH = -arch=sm_70

CXX	= nvcc
CXXFLAGS = --compiler-options "$(OPT) $(PIC) $(OMP)" $(XARCH) $(XOPT) $(XPIC)

INCLUDES = -I$(CUDA_ROOT)/include \
	    -I$(CUDA_ROOT)/samples/NVIDIA_CUDA-10.2_Samples/common/inc

SOFLAGS = -shared
XLIBS	= -lcublas

$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	/bin/rm -f $(TARGET) $(LIBOBJS) 
