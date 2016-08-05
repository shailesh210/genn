
CXXFLAGS       =/nologo /EHsc /c  /O2

INCLUDEFLAGS   =/I"$(GENN_PATH)\lib\include" -I"$(OPENCL_PATH)\include" 

all: runner.obj

runner.obj: runner.cc
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc /link /LIBPATH:"$(OPENCL_PATH)\lib\Win32" OpenCL.lib

clean:
	-del runner.obj 2>nul
