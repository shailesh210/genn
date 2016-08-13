/*--------------------------------------------------------------------------
  Author/Modifier: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
   
  --------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file utils.h

  \brief This file contains standard utility functions provide within the NVIDIA CUDA software development toolkit (SDK). The remainder of the file contains a function that defines the standard neuron models.
*/
//--------------------------------------------------------------------------

#ifndef _UTILS_H_
#define _UTILS_H_ //!< macro for avoiding multiple inclusion during compilation

#include <iostream>
#include <string>

#ifndef CPU_ONLY
	#ifdef OPENCL
		#include <CL/cl.h>
	#else
		#include <cuda.h>
		#include <cuda_runtime.h>
	#endif
#endif	//CPU_ONLY

using namespace std;


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Macros for catching errors returned by the CUDA driver and runtime APIs.
 */
//--------------------------------------------------------------------------
#ifdef OPENCL

#ifndef OpenClErrorCodeToString
#define OpenClErrorCodeToString(errorCode)\
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

	#ifndef CHECK_CL_ERRORS
	#define CHECK_CL_ERRORS(call){\
		if(call != CL_SUCCESS)\
								{\
		OpenClErrorCodeToString(call) ; \
		  cout<<endl;\
			cout << "Location : " << __FILE__ << ":" << __LINE__ << endl; \
			exit(EXIT_FAILURE);\
						}\
		}
	#endif
#else


	#if CUDA_VERSION >= 6050
	#define CHECK_CU_ERRORS(call)					\
	{								\
		CUresult error = call;					\
		if (error != CUDA_SUCCESS)					\
		{								\
		const char *errStr;					\
		cuGetErrorName(error, &errStr);				\
		cerr << __FILE__ << ": " <<  __LINE__;			\
		cerr << ": cuda driver error " << error << ": ";	\
		cerr << errStr << endl;					\
		exit(EXIT_FAILURE);					\
		}								\
	}
	#else
	#define CHECK_CU_ERRORS(call) call
	#endif


// comment below and uncomment here when using CUDA that does not support cugetErrorName
//#define CHECK_CU_ERRORS(call) call
#define CHECK_CUDA_ERRORS(call)					\
  {								\
    cudaError_t error = call;					\
    if (error != cudaSuccess)					\
      {								\
	cerr << __FILE__ << ": " <<  __LINE__;			\
	cerr << ": cuda runtime error " << error << ": ";	\
	cerr << cudaGetErrorString(error) << endl;		\
	exit(EXIT_FAILURE);					\
      }								\
  }
#endif
#endif


//--------------------------------------------------------------------------
/*! \brief Bit tool macros
 */
//--------------------------------------------------------------------------

#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x

#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1

#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Function for getting the capabilities of a CUDA device via the driver API.
 */
//--------------------------------------------------------------------------
#ifndef OPENCL // CUDA
	CUresult cudaFuncGetAttributesDriver(cudaFuncAttributes *attr, CUfunction kern);
#endif
#endif


//--------------------------------------------------------------------------
/*! \brief Function called upon the detection of an error. Outputs an error message and then exits.
 */
//--------------------------------------------------------------------------

void gennError(string error);


//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(string type);


//--------------------------------------------------------------------------
/*! \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &os);


#endif  // _UTILS_H_
