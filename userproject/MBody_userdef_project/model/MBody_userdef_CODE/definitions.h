

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model MBody_userdef containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "utils.h"
#include "hr_time.h"
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
          break;\
    case CL_INVALID_PLATFORM:\
        cout << "CL_INVALID_PLATFORM";\
          break;\
    case CL_INVALID_DEVICE:\
        cout << "CL_INVALID_DEVICE";\
          break;\
    case CL_INVALID_CONTEXT:\
        cout << "CL_INVALID_CONTEXT";\
          break;\
   case CL_INVALID_QUEUE_PROPERTIES:\
       cout << "CL_INVALID_QUEUE_PROPERTIES";\
          break;\
    case CL_INVALID_COMMAND_QUEUE:\
        cout << "CL_INVALID_COMMAND_QUEUE";\
          break;\
    case CL_INVALID_HOST_PTR:\
        cout << "CL_INVALID_HOST_PTR";\
          break;\
    case CL_INVALID_MEM_OBJECT:\
        cout << "CL_INVALID_MEM_OBJECT";\
          break;\
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:\
       cout << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";\
          break;\
    case CL_INVALID_IMAGE_SIZE:\
         cout << "CL_INVALID_IMAGE_SIZE";\
          break;\
    case CL_INVALID_SAMPLER:\
        cout << "CL_INVALID_SAMPLER";\
          break;\
    case CL_INVALID_BINARY:\
        cout << "CL_INVALID_BINARY";\
          break;\
    case CL_INVALID_BUILD_OPTIONS:\
        cout << "CL_INVALID_BUILD_OPTIONS";\
          break;\
    case CL_INVALID_PROGRAM:\
        cout << "CL_INVALID_PROGRAM";\
          break;\
    case CL_INVALID_PROGRAM_EXECUTABLE:\
        cout << "CL_INVALID_PROGRAM_EXECUTABLE";\
          break;\
    case CL_INVALID_KERNEL_NAME:\
        cout << "CL_INVALID_KERNEL_NAME";\
          break;\
    case CL_INVALID_KERNEL_DEFINITION:\
        cout << "CL_INVALID_KERNEL_DEFINITION";\
          break;\
    case CL_INVALID_KERNEL:\
        cout << "CL_INVALID_KERNEL";\
          break;\
    case CL_INVALID_ARG_INDEX:\
        cout << "CL_INVALID_ARG_INDEX";\
          break;\
    case CL_INVALID_ARG_VALUE:\
        cout << "CL_INVALID_ARG_VALUE";\
          break;\
    case CL_INVALID_ARG_SIZE:\
        cout << "CL_INVALID_ARG_SIZE";\
          break;\
    case CL_INVALID_KERNEL_ARGS:\
        cout << "CL_INVALID_KERNEL_ARGS";\
          break;\
    case CL_INVALID_WORK_DIMENSION:\
        cout << "CL_INVALID_WORK_DIMENSION";\
          break;\
    case CL_INVALID_WORK_GROUP_SIZE:\
        cout << "CL_INVALID_WORK_GROUP_SIZE";\
          break;\
    case CL_INVALID_WORK_ITEM_SIZE:\
        cout << "CL_INVALID_WORK_ITEM_SIZE";\
          break;\
    case CL_INVALID_GLOBAL_OFFSET:\
        cout << "CL_INVALID_GLOBAL_OFFSET";\
          break;\
    case CL_INVALID_EVENT_WAIT_LIST:\
        cout << "CL_INVALID_EVENT_WAIT_LIST";\
          break;\
    case CL_INVALID_EVENT:\
        cout << "CL_INVALID_EVENT";\
          break;\
    case CL_INVALID_OPERATION:\
       cout << "CL_INVALID_OPERATION";\
          break;\
    case CL_INVALID_GL_OBJECT:\
        cout << "CL_INVALID_GL_OBJECT";\
          break;\
    case CL_INVALID_BUFFER_SIZE:\
        cout << "CL_INVALID_BUFFER_SIZE";\
          break;\
    case CL_INVALID_MIP_LEVEL:\
       cout << "CL_INVALID_MIP_LEVEL";\
          break;\
    case CL_INVALID_GLOBAL_WORK_SIZE:\
        cout << "CL_INVALID_GLOBAL_WORK_SIZE";\
          break;\
    case CL_PLATFORM_NOT_FOUND_KHR:\
        cout << "CL_PLATFORM_NOT_FOUND_KHR";\
          break;\
    case CL_DEVICE_PARTITION_FAILED_EXT:\
        cout << "CL_DEVICE_PARTITION_FAILED_EXT";\
          break;\
    case CL_INVALID_PARTITION_COUNT_EXT:\
        cout << "CL_INVALID_PARTITION_COUNT_EXT";\
          break;\
    default:\
        cout << "unknown error code";\
          break;\
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
#define DT 0.100000f
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
extern cl_event neuronStart, neuronStop;
extern cl_event synapseStart, synapseStop;
extern double synapse_tme;
extern CStopWatch synapse_timer;
extern cl_event learningStart, learningStop;
extern double learning_tme;
extern CStopWatch learning_timer;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntPN;
extern cl_mem d_glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern cl_mem d_glbSpkPN;
extern float * VPN;
extern cl_mem d_VPN;
extern uint64_t * seedPN;
extern cl_mem d_seedPN;
extern float * spikeTimePN;
extern cl_mem d_spikeTimePN;
extern uint64_t * ratesPN;
extern cl_mem d_ratesPN;
extern unsigned int offsetPN;
extern cl_mem d_offsetPN;
extern unsigned int * glbSpkCntKC;
extern cl_mem d_glbSpkCntKC;
extern unsigned int * glbSpkKC;
extern cl_mem d_glbSpkKC;
extern float * sTKC;
extern cl_mem d_sTKC;
extern float * VKC;
extern cl_mem d_VKC;
extern float * mKC;
extern cl_mem d_mKC;
extern float * hKC;
extern cl_mem d_hKC;
extern float * nKC;
extern cl_mem d_nKC;
extern unsigned int * glbSpkCntLHI;
extern cl_mem d_glbSpkCntLHI;
extern unsigned int * glbSpkLHI;
extern cl_mem d_glbSpkLHI;
extern unsigned int * glbSpkCntEvntLHI;
extern cl_mem d_glbSpkCntEvntLHI;
extern unsigned int * glbSpkEvntLHI;
extern cl_mem d_glbSpkEvntLHI;
extern float * VLHI;
extern cl_mem d_VLHI;
extern float * mLHI;
extern cl_mem d_mLHI;
extern float * hLHI;
extern cl_mem d_hLHI;
extern float * nLHI;
extern cl_mem d_nLHI;
extern unsigned int * glbSpkCntDN;
extern cl_mem d_glbSpkCntDN;
extern unsigned int * glbSpkDN;
extern cl_mem d_glbSpkDN;
extern unsigned int * glbSpkCntEvntDN;
extern cl_mem d_glbSpkCntEvntDN;
extern unsigned int * glbSpkEvntDN;
extern cl_mem d_glbSpkEvntDN;
extern float * sTDN;
extern cl_mem d_sTDN;
extern float * VDN;
extern cl_mem d_VDN;
extern float * mDN;
extern cl_mem d_mDN;
extern float * hDN;
extern cl_mem d_hDN;
extern float * nDN;
extern cl_mem d_nDN;

#define glbSpkShiftPN 0
#define glbSpkShiftKC 0
#define glbSpkShiftLHI 0
#define glbSpkShiftDN 0
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN
#define spikeCount_KC glbSpkCntKC[0]
#define spike_KC glbSpkKC
#define spikeCount_LHI glbSpkCntLHI[0]
#define spike_LHI glbSpkLHI
#define spikeEventCount_LHI glbSpkCntEvntLHI[0]
#define spikeEvent_LHI glbSpkEvntLHI
#define spikeCount_DN glbSpkCntDN[0]
#define spike_DN glbSpkDN
#define spikeEventCount_DN glbSpkCntEvntDN[0]
#define spikeEvent_DN glbSpkEvntDN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynPNKC;
extern cl_mem d_inSynPNKC;
extern SparseProjection CPNKC;
extern float * gPNKC;
extern cl_mem d_gPNKC;
extern float * EEEEPNKC;
extern cl_mem d_EEEEPNKC;
extern float * inSynPNLHI;
extern cl_mem d_inSynPNLHI;
extern float * gPNLHI;
extern cl_mem d_gPNLHI;
extern float * inSynLHIKC;
extern cl_mem d_inSynLHIKC;
extern float * inSynKCDN;
extern cl_mem d_inSynKCDN;
extern SparseProjection CKCDN;
extern float * gKCDN;
extern cl_mem d_gKCDN;
extern float * gRawKCDN;
extern cl_mem d_gRawKCDN;
extern float * inSynDNDN;
extern cl_mem d_inSynDNDN;

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

void pushPNStateToDevice();
void pushPNSpikesToDevice();
void pushPNSpikeEventsToDevice();
void pushPNCurrentSpikesToDevice();
void pushPNCurrentSpikeEventsToDevice();
void pushKCStateToDevice();
void pushKCSpikesToDevice();
void pushKCSpikeEventsToDevice();
void pushKCCurrentSpikesToDevice();
void pushKCCurrentSpikeEventsToDevice();
void pushLHIStateToDevice();
void pushLHISpikesToDevice();
void pushLHISpikeEventsToDevice();
void pushLHICurrentSpikesToDevice();
void pushLHICurrentSpikeEventsToDevice();
void pushDNStateToDevice();
void pushDNSpikesToDevice();
void pushDNSpikeEventsToDevice();
void pushDNCurrentSpikesToDevice();
void pushDNCurrentSpikeEventsToDevice();
#define pushPNKCToDevice pushPNKCStateToDevice
void pushPNKCStateToDevice();
#define pushPNLHIToDevice pushPNLHIStateToDevice
void pushPNLHIStateToDevice();
#define pushLHIKCToDevice pushLHIKCStateToDevice
void pushLHIKCStateToDevice();
#define pushKCDNToDevice pushKCDNStateToDevice
void pushKCDNStateToDevice();
#define pushDNDNToDevice pushDNDNStateToDevice
void pushDNDNStateToDevice();

// ------------------------------------------------------------------------
// copying things from device

void pullPNStateFromDevice();
void pullPNSpikesFromDevice();
void pullPNSpikeEventsFromDevice();
void pullPNCurrentSpikesFromDevice();
void pullPNCurrentSpikeEventsFromDevice();
void pullKCStateFromDevice();
void pullKCSpikesFromDevice();
void pullKCSpikeEventsFromDevice();
void pullKCCurrentSpikesFromDevice();
void pullKCCurrentSpikeEventsFromDevice();
void pullLHIStateFromDevice();
void pullLHISpikesFromDevice();
void pullLHISpikeEventsFromDevice();
void pullLHICurrentSpikesFromDevice();
void pullLHICurrentSpikeEventsFromDevice();
void pullDNStateFromDevice();
void pullDNSpikesFromDevice();
void pullDNSpikeEventsFromDevice();
void pullDNCurrentSpikesFromDevice();
void pullDNCurrentSpikeEventsFromDevice();
#define pullPNKCFromDevice pullPNKCStateFromDevice
void pullPNKCStateFromDevice();
#define pullPNLHIFromDevice pullPNLHIStateFromDevice
void pullPNLHIStateFromDevice();
#define pullLHIKCFromDevice pullLHIKCStateFromDevice
void pullLHIKCStateFromDevice();
#define pullKCDNFromDevice pullKCDNStateFromDevice
void pullKCDNStateFromDevice();
#define pullDNDNFromDevice pullDNDNStateFromDevice
void pullDNDNStateFromDevice();

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
// Function for mapping OPENCL buffers.

void mapBuffer();

// ------------------------------------------------------------------------
// Function for unmapping OPENCL buffers.

void unmapBuffer();

// ------------------------------------------------------------------------
// Function for setting OPENCL kernel arguments.

void set_kernel_arguments();

// ------------------------------------------------------------------------
// Function for setting the CUDA device and the host's global variables.
// Also estimates memory usage on device.

void allocateMem();

void allocatePNKC(unsigned int connN);

void allocateKCDN(unsigned int connN);

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

void initializeAllSparseArrays();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initMBody_userdef();

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
