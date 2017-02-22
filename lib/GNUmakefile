##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##              Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------


# Makefile for the GeNN "generateALL" and "libgenn.a" binaries
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit)
OS_SIZE                      :=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_UPPER                     :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                     :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                       :=$(strip $(findstring DARWIN,$(OS_UPPER)))

# Source, Include, Object and Library directories
SRC_PATH                     :=$(GENN_PATH)/lib/src
INC_PATH                     :=$(GENN_PATH)/lib/include
GENERATEALL_PATH             :=$(CURDIR)
LIBGENN_PATH                 :=$(GENN_PATH)/lib/lib
ifndef CPU_ONLY
    ifdef OPENCL
        LIBGENN_OBJ_PATH     :=$(GENN_PATH)/lib/obj_OPENCL
    else
        LIBGENN_OBJ_PATH     :=$(GENN_PATH)/lib/obj
    endif
else
    LIBGENN_OBJ_PATH         :=$(GENN_PATH)/lib/obj_CPU_ONLY
endif

# generateALL and libgenn.a targets
ifndef CPU_ONLY
    ifdef OPENCL
        GENERATEALL          :=$(GENERATEALL_PATH)/generateALL_OPENCL
        LIBGENN              :=$(LIBGENN_PATH)/libgenn_OPENCL.a
    else
        GENERATEALL          :=$(GENERATEALL_PATH)/generateALL
        LIBGENN              :=$(LIBGENN_PATH)/libgenn.a
    endif
else
    GENERATEALL              :=$(GENERATEALL_PATH)/generateALL_CPU_ONLY
    LIBGENN                  :=$(LIBGENN_PATH)/libgenn_CPU_ONLY.a
endif
LIBGENN_OBJ                  :=global.o modelSpec.o neuronModels.o synapseModels.o postSynapseModels.o utils.o stringUtils.o sparseUtils.o hr_time.o
LIBGENN_OBJ                  :=$(addprefix $(LIBGENN_OBJ_PATH)/,$(LIBGENN_OBJ))

# Global CUDA compiler settings
ifndef CPU_ONLY
    OPENCL_PATH              ?=/opt/intel/opencl
    CUDA_PATH                ?=/usr/local/cuda
    NVCC                     :="$(CUDA_PATH)/bin/nvcc"
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX                      :=clang++
endif
ifndef CPU_ONLY
    ifdef OPENCL
        CXXFLAGS                 :=-std=c++11 -DOPENCL
    else
        CXXFLAGS                 :=-std=c++11 -DNVCC=\"$(NVCC)\"
    endif
else
    CXXFLAGS                 :=-std=c++11 -DCPU_ONLY
endif
ifdef DEBUG
    CXXFLAGS                 +=-g -O0 -DDEBUG
endif
ARFLAGS                      :=-rcs

# Global include and link flags
ifndef CPU_ONLY
    ifdef OPENCL
        INCLUDE_FLAGS            :=-I"$(INC_PATH)" -I"$(OPENCL_PATH)/include"
    else
        INCLUDE_FLAGS            :=-I"$(INC_PATH)" -I"$(CUDA_PATH)/include"
    endif
    ifeq ($(DARWIN),DARWIN)
        ifdef OPENCL
            LINK_FLAGS       :=-L"$(LIBGENN_PATH)" -L"$(OPENCL_PATH)" -lgenn_OPENCL -lOpenCL -lstdc++ -lc++
        else
            LINK_FLAGS       :=-L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart -lstdc++ -lc++
        endif
    else
        ifeq ($(OS_SIZE),32)
            ifdef OPENCL
                LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(OPENCL_PATH)" -lgenn_OPENCL -lOpenCL
            else
                LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart
            endif
        else
            ifdef OPENCL
                LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(OPENCL_PATH)" -lgenn_OPENCL -lOpenCL
            else
                LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib64" -lgenn -lcuda -lcudart
            endif
        endif
    endif
else
    INCLUDE_FLAGS            :=-I"$(INC_PATH)"
    LINK_FLAGS               :=-L"$(LIBGENN_PATH)" -lgenn_CPU_ONLY
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS           :=-L"$(LIBGENN_PATH)" -lgenn_CPU_ONLY -lstdc++ -lc++
    endif
endif


# Target rules
.PHONY: all clean clean_generateall clean_libgenn always_compile

all: $(GENERATEALL)

$(GENERATEALL): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -DMODEL=\"$(MODEL)\" -o $@ $(SRC_PATH)/generate*.cc $(INCLUDE_FLAGS) $(LINK_FLAGS)

$(LIBGENN): $(LIBGENN_OBJ_PATH) $(LIBGENN_OBJ) $(LIBGENN_PATH)
	$(AR) $(ARFLAGS) $@ $(LIBGENN_OBJ)

$(LIBGENN_OBJ_PATH)/%.o: $(SRC_PATH)/%.cc
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $< $(INCLUDE_FLAGS)

$(LIBGENN_OBJ_PATH):
	mkdir -p $@

$(LIBGENN_PATH):
	mkdir -p $@

clean: clean_generateall clean_libgenn

clean_generateall:
	rm -f $(GENERATEALL) $(GENERATEALL).d

clean_libgenn:
	rm -rf $(LIBGENN_OBJ_PATH) $(LIBGENN_PATH)

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))
