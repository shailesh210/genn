
NVCC           ="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/bin/nvcc.exe"
NVCCFLAGS      =-c -x cu -arch sm_12 -O3 -use_fast_math

INCLUDEFLAGS   =-I"$(GENN_PATH)\lib\include"

all: runner.obj

runner.obj: runner.cc
	$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc

clean:
	-del runner.obj 2>nul
