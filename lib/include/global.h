/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file global.h

\brief Global header file containing a few global variables. Part of the code generation section.
*/
//--------------------------------------------------------------------------

#ifndef GLOBAL_H
#define GLOBAL_H


#define OPENCL


#ifndef CPU_ONLY
	#ifdef OPENCL
		#include <CL/cl.h>
	#else
		#include <cuda.h>
		#include <cuda_runtime.h>
	#endif
#endif


namespace GENN_FLAGS {
    extern unsigned int calcSynapseDynamics;
    extern unsigned int calcSynapses;
    extern unsigned int learnSynapsesPost;
    extern unsigned int calcNeurons;
};

namespace GENN_PREFERENCES {    
    extern int optimiseBlockSize; //!< Flag for signalling whether or not block size optimisation should be performed
    extern int autoChooseDevice; //!< Flag to signal whether the GPU device should be chosen automatically 
    extern bool optimizeCode; //!< Request speed-optimized code, at the expense of floating-point accuracy
    extern bool debugCode; //!< Request debug data to be embedded in the generated code
    extern bool showPtxInfo; //!< Request that PTX assembler information be displayed for each CUDA kernel during compilation
    extern double asGoodAsZero; //!< Global variable that is used when detecting close to zero values, for example when setting sparse connectivity from a dense matrix
    extern int defaultDevice; //! default GPU device; used to determine which GPU to use if chooseDevice is 0 (off)
    extern unsigned int neuronBlockSize;
    extern unsigned int synapseBlockSize;
    extern unsigned int learningBlockSize;
    extern unsigned int synapseDynamicsBlockSize;
    extern unsigned int autoRefractory; //!< Flag for signalling whether spikes are only reported if thresholdCondition changes from false to true (autoRefractory == 1) or spikes are emitted whenever thresholdCondition is true no matter what.
};

#ifdef OPENCL
	struct CLDeviceProp{
		int MAX_WORK_GROUP_SIZE;							//maxThreadsPerBlock
		unsigned long DEVICE_LOCAL_MEM_SIZE;						//sharedMemPerBlock
		unsigned int DEVICE_MAX_COMPUTE_UNITS;					//multiProcessorCount
		char DEVICE_NAME[1000];									//name
		unsigned long DEVICE_GLOBAL_MEM_SIZE;					//totalGlobalMem
		unsigned int REGISTERSS_PER_BLOCK;						//regsPerBlock
		unsigned int MAX_WORK_UNITS_PER_COMPUTE_UNIT;			//maxThreadsPerMultiProcessor      CHECK THIS LATER	
		int major;												//device version major
		int minor;												//device version minor
};
#endif  //OPENCL



extern int neuronBlkSz; //!< Global variable containing the GPU block size for the neuron kernel
extern int synapseBlkSz; //!< Global variable containing the GPU block size for the synapse kernel
extern int learnBlkSz; //!< Global variable containing the GPU block size for the learn kernel
extern int synDynBlkSz; //!< Global variable containing the GPU block size for the synapse dynamics kernel
//extern vector<cudaDeviceProp> deviceProp; //!< Global vector containing the properties of all CUDA-enabled devices
//extern vector<int> synapseBlkSz; //!< Global vector containing the optimum synapse kernel block size for each device
//extern vector<int> learnBlkSz; //!< Global vector containing the optimum learn kernel block size for each device
//extern vector<int> neuronBlkSz; //!< Global vector containing the optimum neuron kernel block size for each device
//extern vector<int> synDynBlkSz; //!< Global vector containing the optimum synapse dynamics kernel block size for each device
#ifndef CPU_ONLY
	#ifdef OPENCL
		extern CLDeviceProp *deviceProp;
		extern int theDevice;		//!< Global variable containing the currently selected OPENCL device's number
		extern int deviceCount;	//!< Global variable containing the number of OPENCL devices on this host
		extern cl_device_id device_ids[100];		//array of device ids ; for now 100...try replace by num_of _devices
		extern cl_platform_id platform_id;  //!< Global variables platform_id
		extern cl_uint ret_num_platforms;			//!< Global variables number of platforms
	
	#else
		extern cudaDeviceProp *deviceProp;
		extern int theDevice; //!< Global variable containing the currently selected CUDA device's number
		extern int deviceCount; //!< Global variable containing the number of CUDA devices on this host
		
			
	#endif  // OPENCL
#endif


extern int hostCount; //!< Global variable containing the number of hosts within the local compute cluster

#endif // GLOBAL_H

#ifndef TRUE
#define TRUE true
#endif

#ifndef FALSE
#define FALSE false
#endif
