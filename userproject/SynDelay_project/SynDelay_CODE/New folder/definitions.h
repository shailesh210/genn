

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model SynDelay containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "utils.h"
#include "sparseUtils.h"

#include "sparseProjection.h"
#include <stdint.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifndef OpenCLErrorCodeToString
#define OpenCLErrorCodeToString(errorCode)\
{\
	switch(errorCode)\
	{\
  case CL_INVALID_DEVICE_TYPE:\
        cout << "CL_INVALID_DEVICE_TYPE";\
    case CL_INVALID_PLATFORM:\
        cout << "CL_INVALID_PLATFORM";\
    case CL_INVALID_DEVICE:\
        cout << "CL_INVALID_DEVICE";\
    case CL_INVALID_CONTEXT:\
        cout << "CL_INVALID_CONTEXT";\
   case CL_INVALID_QUEUE_PROPERTIES:\
       cout << "CL_INVALID_QUEUE_PROPERTIES";\
    case CL_INVALID_COMMAND_QUEUE:\
        cout << "CL_INVALID_COMMAND_QUEUE";\
    case CL_INVALID_HOST_PTR:\
        cout << "CL_INVALID_HOST_PTR";\
    case CL_INVALID_MEM_OBJECT:\
        cout << "CL_INVALID_MEM_OBJECT";\
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:\
       cout << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";\
    case CL_INVALID_IMAGE_SIZE:\
         cout << "CL_INVALID_IMAGE_SIZE";\
    case CL_INVALID_SAMPLER:\
        cout << "CL_INVALID_SAMPLER";\
    case CL_INVALID_BINARY:\
        cout << "CL_INVALID_BINARY";\
    case CL_INVALID_BUILD_OPTIONS:\
        cout << "CL_INVALID_BUILD_OPTIONS";\
    case CL_INVALID_PROGRAM:\
        cout << "CL_INVALID_PROGRAM";\
    case CL_INVALID_PROGRAM_EXECUTABLE:\
        cout << "CL_INVALID_PROGRAM_EXECUTABLE";\
    case CL_INVALID_KERNEL_NAME:\
        cout << "CL_INVALID_KERNEL_NAME";\
    case CL_INVALID_KERNEL_DEFINITION:\
        cout << "CL_INVALID_KERNEL_DEFINITION";\
    case CL_INVALID_KERNEL:\
        cout << "CL_INVALID_KERNEL";\
    case CL_INVALID_ARG_INDEX:\
        cout << "CL_INVALID_ARG_INDEX";\
    case CL_INVALID_ARG_VALUE:\
        cout << "CL_INVALID_ARG_VALUE";\
    case CL_INVALID_ARG_SIZE:\
        cout << "CL_INVALID_ARG_SIZE";\
    case CL_INVALID_KERNEL_ARGS:\
        cout << "CL_INVALID_KERNEL_ARGS";\
    case CL_INVALID_WORK_DIMENSION:\
        cout << "CL_INVALID_WORK_DIMENSION";\
    case CL_INVALID_WORK_GROUP_SIZE:\
        cout << "CL_INVALID_WORK_GROUP_SIZE";\
    case CL_INVALID_WORK_ITEM_SIZE:\
        cout << "CL_INVALID_WORK_ITEM_SIZE";\
    case CL_INVALID_GLOBAL_OFFSET:\
        cout << "CL_INVALID_GLOBAL_OFFSET";\
    case CL_INVALID_EVENT_WAIT_LIST:\
        cout << "CL_INVALID_EVENT_WAIT_LIST";\
    case CL_INVALID_EVENT:\
        cout << "CL_INVALID_EVENT";\
    case CL_INVALID_OPERATION:\
       cout << "CL_INVALID_OPERATION";\
    case CL_INVALID_GL_OBJECT:\
        cout << "CL_INVALID_GL_OBJECT";\
    case CL_INVALID_BUFFER_SIZE:\
        cout << "CL_INVALID_BUFFER_SIZE";\
    case CL_INVALID_MIP_LEVEL:\
       cout << "CL_INVALID_MIP_LEVEL";\
    case CL_INVALID_GLOBAL_WORK_SIZE:\
        cout << "CL_INVALID_GLOBAL_WORK_SIZE";\
    case CL_PLATFORM_NOT_FOUND_KHR:\
        cout << "CL_PLATFORM_NOT_FOUND_KHR";\
    case CL_DEVICE_PARTITION_FAILED_EXT:\
        cout << "CL_DEVICE_PARTITION_FAILED_EXT";\
    case CL_INVALID_PARTITION_COUNT_EXT:\
        cout << "CL_INVALID_PARTITION_COUNT_EXT";\
    default:\
        cout << "unknown error code";\
}\
}
#endif
#ifndef CHECK_OPENCL_ERRORS
#define CHECK_OPENCL_ERRORS(call){\
		if(call != CL_SUCCESS)\
		{\
			OpenCLErrorCodeToString(call) ; \
           cout << endl;\
			cout << "Location : " << __FILE__ << ":" << __LINE__ << endl; \
			exit(EXIT_FAILURE);\
		}\
}
#endif
#undef DT
#define DT 1.00000f
#ifndef MYRAND
#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);
#endif
#ifndef MYRAND_MAX
#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL
#endif

#ifndef MEM_SIZE
#define MEM_SIZE (1024)
#endif
#ifndef MAX_SOURCE_SIZE
#define MAX_SOURCE_SIZE (0x110000)
#endif
#ifndef scalar
typedef float scalar;
#endif
#ifndef SCALAR_MIN
#define SCALAR_MIN 1.17549e-038f
#endif
#ifndef SCALAR_MAX
#define SCALAR_MAX 3.40282e+038f
#endif

// ------------------------------------------------------------------------
// global variables

extern unsigned long long iT;
extern float t;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntInput;
extern cl_mem d_glbSpkCntInput;
extern unsigned int * glbSpkInput;
extern cl_mem d_glbSpkInput;
extern unsigned int spkQuePtrInput;
extern float * VInput;
extern cl_mem d_VInput;
extern float * UInput;
extern cl_mem d_UInput;
extern unsigned int * glbSpkCntInter;
extern cl_mem d_glbSpkCntInter;
extern unsigned int * glbSpkInter;
extern cl_mem d_glbSpkInter;
extern float * VInter;
extern cl_mem d_VInter;
extern float * UInter;
extern cl_mem d_UInter;
extern unsigned int * glbSpkCntOutput;
extern cl_mem d_glbSpkCntOutput;
extern unsigned int * glbSpkOutput;
extern cl_mem d_glbSpkOutput;
extern float * VOutput;
extern cl_mem d_VOutput;
extern float * UOutput;
extern cl_mem d_UOutput;

#define glbSpkShiftInput spkQuePtrInput*500
#define glbSpkShiftInter 0
#define glbSpkShiftOutput 0
#define spikeCount_Input glbSpkCntInput[spkQuePtrInput]
#define spike_Input (glbSpkInput+(spkQuePtrInput*500))
#define spikeCount_Inter glbSpkCntInter[0]
#define spike_Inter glbSpkInter
#define spikeCount_Output glbSpkCntOutput[0]
#define spike_Output glbSpkOutput

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynInputInter;
extern cl_mem d_inSynInputInter;
extern float * inSynInputOutput;
extern cl_mem d_inSynInputOutput;
extern float * inSynInterOutput;
extern cl_mem d_inSynInterOutput;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// Helper function for allocating memory blocks on the GPU device

template<class T>
void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)
{
    void *devptr;
    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));
    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));
    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// copying things to device

void pushInputStateToDevice();
void pushInputSpikesToDevice();
void pushInputSpikeEventsToDevice();
void pushInputCurrentSpikesToDevice();
void pushInputCurrentSpikeEventsToDevice();
void pushInterStateToDevice();
void pushInterSpikesToDevice();
void pushInterSpikeEventsToDevice();
void pushInterCurrentSpikesToDevice();
void pushInterCurrentSpikeEventsToDevice();
void pushOutputStateToDevice();
void pushOutputSpikesToDevice();
void pushOutputSpikeEventsToDevice();
void pushOutputCurrentSpikesToDevice();
void pushOutputCurrentSpikeEventsToDevice();
#define pushInputInterToDevice pushInputInterStateToDevice
void pushInputInterStateToDevice();
#define pushInputOutputToDevice pushInputOutputStateToDevice
void pushInputOutputStateToDevice();
#define pushInterOutputToDevice pushInterOutputStateToDevice
void pushInterOutputStateToDevice();

// ------------------------------------------------------------------------
// copying things from device

void pullInputStateFromDevice();
void pullInputSpikesFromDevice();
void pullInputSpikeEventsFromDevice();
void pullInputCurrentSpikesFromDevice();
void pullInputCurrentSpikeEventsFromDevice();
void pullInterStateFromDevice();
void pullInterSpikesFromDevice();
void pullInterSpikeEventsFromDevice();
void pullInterCurrentSpikesFromDevice();
void pullInterCurrentSpikeEventsFromDevice();
void pullOutputStateFromDevice();
void pullOutputSpikesFromDevice();
void pullOutputSpikeEventsFromDevice();
void pullOutputCurrentSpikesFromDevice();
void pullOutputCurrentSpikeEventsFromDevice();
#define pullInputInterFromDevice pullInputInterStateFromDevice
void pullInputInterStateFromDevice();
#define pullInputOutputFromDevice pullInputOutputStateFromDevice
void pullInputOutputStateFromDevice();
#define pullInterOutputFromDevice pullInterOutputStateFromDevice
void pullInterOutputStateFromDevice();

// ------------------------------------------------------------------------
// global copying values to device

void copyStateToDevice();

// ------------------------------------------------------------------------
// global copying spikes to device

void copySpikesToDevice();

// ------------------------------------------------------------------------
// copying current spikes to device

void copyCurrentSpikesToDevice();

// ------------------------------------------------------------------------
// global copying spike events to device

void copySpikeEventsToDevice();

// ------------------------------------------------------------------------
// copying current spikes to device

void copyCurrentSpikeEventsToDevice();

// ------------------------------------------------------------------------
// global copying values from device

void copyStateFromDevice();

// ------------------------------------------------------------------------
// global copying spikes from device

void copySpikesFromDevice();

// ------------------------------------------------------------------------
// copying current spikes from device

void copyCurrentSpikesFromDevice();

// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)

void copySpikeNFromDevice();

// ------------------------------------------------------------------------
// global copying spikeEvents from device

void copySpikeEventsFromDevice();

// ------------------------------------------------------------------------
// copying current spikeEvents from device

void copyCurrentSpikeEventsFromDevice();

// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)

void copySpikeEventNFromDevice();

// ------------------------------------------------------------------------
// Function for setting the CUDA device and the host's global variables.
// Also estimates memory usage on device.

void mapBuffer();

// ------------------------------------------------------------------------
// Function for setting the CUDA device and the host's global variables.
// Also estimates memory usage on device.

void allocateMem();

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

void initializeAllSparseArrays();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initSynDelay();

// ------------------------------------------------------------------------
// Function to free all global memory structures.

void freeMem();

//-------------------------------------------------------------------------
// Function to convert a firing probability (per time step) to an integer of type uint64_t
// that can be used as a threshold for the GeNN random number generator to generate events with the given probability.

void convertProbabilityToRandomNumberThreshold(float *p_pattern, uint64_t *pattern, int N);

//-------------------------------------------------------------------------
// Function to convert a firing rate (in kHz) to an integer of type uint64_t that can be used
// as a threshold for the GeNN random number generator to generate events with the given rate.

void convertRateToRandomNumberThreshold(float *rateKHz_pattern, uint64_t *pattern, int N);

// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls (using CPU)

template <class T>
void stepTimeCPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)

void stepTimeCPU();

// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls (using GPU)

template <class T>
void stepTimeGPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

// ------------------------------------------------------------------------
// the actual time stepping procedure (using GPU)

void stepTimeGPU();

#endif
