GPU=0
STOCHASTIC=1
FIXEDPOINT=0

VPATH=./src/ ./src/data
OBJDIR=./obj/
EXEC=nn

CC=gcc
CPP=g++

COMMON=-Isrc/ -Isrc/data/ -DSTOCHASTIC=$(STOCHASTIC) -DFIXEDPOINT=$(FIXEDPOINT)
CFLAGS=-fopenmp -O3 -std=c++11
LDFLAGS=-fopenmp

ifeq ($(GPU), 1)
NVCC=nvcc
ARCH=-gencode arch=compute_30,code=sm_30 \
     -gencode arch=compute_35,code=sm_35 \
     -gencode arch=compute_50,code=[sm_50,compute_50] \
     -gencode arch=compute_52,code=[sm_52,compute_52]
COMMON+=-DGPU=1 -I/usr/local/cuda/include/ -I/usr/local/cuda/samples/common/inc/
NVCCFLAGS=-std=c++11
LDFLAGS+=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++ 
endif

OBJ=Tensor.o Params.o MemoryMonitor.o gemm.o blas.o config.o args.o conv.o
OBJ+=mnist.o svhn.o cifar10.o debug.o
OBJ+=Layer.o LayerAct.o LayerBN.o LayerConv.o LayerDrop.o LayerFull.o LayerInput.o LayerPool.o Net.o main.o
ifeq ($(GPU), 1) 
OBJ+=cuda_util.o blas_gpu.o conv_gpu.o
endif

EXECOBJ=$(addprefix $(OBJDIR), $(OBJ))
DEPS=$(wildcard src/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(EXECOBJ)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

ifeq ($(GPU), 1)
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) -c $< -o $@
endif

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) $(OBJDIR)/*
