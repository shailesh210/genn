

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model SynDelay containing general control code.
*/
//-------------------------------------------------------------------------

#define RUNNER_CC_COMPILE

#include "definitions.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cassert>
#include <stdint.h>

// ------------------------------------------------------------------------
// global variables

unsigned long long iT= 0;
float t;

// ------------------------------------------------------------------------
// neuron variables

cl_mem d_done;
cl_mem d_t;
unsigned int * glbSpkCntInput;
cl_mem d_glbSpkCntInput;
unsigned int * glbSpkInput;
cl_mem d_glbSpkInput;
unsigned int *spkQuePtrInput;
cl_mem d_spkQuePtrInput;
float * VInput;
cl_mem d_VInput;
float * UInput;
cl_mem d_UInput;
unsigned int * glbSpkCntInter;
cl_mem d_glbSpkCntInter;
unsigned int * glbSpkInter;
cl_mem d_glbSpkInter;
float * VInter;
cl_mem d_VInter;
float * UInter;
cl_mem d_UInter;
unsigned int * glbSpkCntOutput;
cl_mem d_glbSpkCntOutput;
unsigned int * glbSpkOutput;
cl_mem d_glbSpkOutput;
float * VOutput;
cl_mem d_VOutput;
float * UOutput;
cl_mem d_UOutput;

// ------------------------------------------------------------------------
// synapse variables

float * inSynInputInter;
cl_mem d_inSynInputInter;
float * inSynInputOutput;
cl_mem d_inSynInputOutput;
float * inSynInterOutput;
cl_mem d_inSynInterOutput;

cl_device_id device_ids[100];
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
cl_kernel calcNeurons = NULL;
cl_kernel calcSynapses = NULL;
cl_kernel learnSynapsesPost = NULL;
cl_kernel calcSynapseDynamics = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;
//-------------------------------------------------------------------------
/*! \brief Function to convert a firing probability (per time step) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability.
*/
//-------------------------------------------------------------------------

void convertProbabilityToRandomNumberThreshold(float *p_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16);
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (p_pattern[i]*fac);
    }
}

//-------------------------------------------------------------------------
/*! \brief Function to convert a firing rate (in kHz) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate.
*/
//-------------------------------------------------------------------------

void convertRateToRandomNumberThreshold(float *rateKHz_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16)*DT;
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (rateKHz_pattern[i]*fac);
    }
}

#include "runnerGPU.cc"

#include "neuronFnct.cc"
#include "synapseFnct.cc"
void allocateMem()
{
/* Get Platform and Device Info */
ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
CHECK_OPENCL_ERRORS(ret);
ret |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, device_ids, &ret_num_devices);
CHECK_OPENCL_ERRORS(ret);
/* Create OpenCL context */
context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
/* Create Command Queue */
command_queue = clCreateCommandQueue(context, device_ids[0], 0, &ret);
CHECK_OPENCL_ERRORS(ret);
FILE *fp;
char fileName1[] = "./SynDelay_CODE/neuronKrnl.cl";
char fileName2[] = "./SynDelay_CODE/synapseKrnl.cl";
char *source_str;
size_t source_size;
/* Load the source code containing the kernel*/
fopen_s(&fp, fileName1, "r");
if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
}
source_str = (char*)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
fclose(fp);
CHECK_OPENCL_ERRORS(ret);
/* Create Kernel Program from the source */
program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
CHECK_OPENCL_ERRORS(ret);
size_t len = 0;
/* Build Kernel Program */
ret |= clBuildProgram(program, 1, device_ids + 0, NULL, NULL, NULL);
ret = clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
char *buffer = (char*)calloc(len, sizeof(char));              // to debug print log
ret = clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
cout << buffer;
CHECK_OPENCL_ERRORS(ret);
/* Create OpenCL Kernel calcNeurons*/
calcNeurons = clCreateKernel(program, "calcNeurons", &ret);
CHECK_OPENCL_ERRORS(ret);
fp=NULL;
source_str=NULL;
/* Load the source code containing the kernel*/
fopen_s(&fp, fileName2, "r");
if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
}
source_str = (char*)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
fclose(fp);
CHECK_OPENCL_ERRORS(ret);
/* Create Kernel Program from the source */
program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
CHECK_OPENCL_ERRORS(ret);
 len = 0;
/* Build Kernel Program */
ret |= clBuildProgram(program, 1, device_ids + 0, NULL, NULL, NULL);
ret = clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
char *buffer2 = (char*)calloc(len, sizeof(char));              // to debug print log
ret = clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, len, buffer2, NULL);
cout << buffer2;
CHECK_OPENCL_ERRORS(ret);
calcSynapses = clCreateKernel(program, "calcSynapses", &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkCntInput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,7 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkInput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,3500 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_spkQuePtrInput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR , sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VInput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_UInput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_glbSpkCntInter=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkInter=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VInter=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_UInter=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_glbSpkCntOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_UOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynInputInter=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynInputOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynInterOutput=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_done =clCreateBuffer(context,  CL_MEM_READ_WRITE ,500 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_t =clCreateBuffer(context,  CL_MEM_READ_WRITE ,500 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_t,CL_TRUE, 0, sizeof(float),(void *)&t,0, NULL, NULL));
}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) time(NULL));

    // neuron variables
//map spkQuePtrInput
spkQuePtrInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_spkQuePtrInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
    spkQuePtrInput[0] = 0;
//unmap spkQuePtrInput
clEnqueueUnmapMemObject(command_queue, d_spkQuePtrInput,spkQuePtrInput, 0, NULL, NULL);
    for (int i = 0; i < 7; i++) {
        glbSpkCntInput[i] = 0;
    }
    for (int i = 0; i < 3500; i++) {
        glbSpkInput[i] = 0;
    }
    for (int i = 0; i < 500; i++) {
        VInput[i] = -65.0000f;
    }
    for (int i = 0; i < 500; i++) {
        UInput[i] = -20.0000f;
    }
    glbSpkCntInter[0] = 0;
    for (int i = 0; i < 500; i++) {
        glbSpkInter[i] = 0;
    }
    for (int i = 0; i < 500; i++) {
        VInter[i] = -65.0000f;
    }
    for (int i = 0; i < 500; i++) {
        UInter[i] = -20.0000f;
    }
    glbSpkCntOutput[0] = 0;
    for (int i = 0; i < 500; i++) {
        glbSpkOutput[i] = 0;
    }
    for (int i = 0; i < 500; i++) {
        VOutput[i] = -65.0000f;
    }
    for (int i = 0; i < 500; i++) {
        UOutput[i] = -20.0000f;
    }

    // synapse variables
    for (int i = 0; i < 500; i++) {
        inSynInputInter[i] = 0.000000f;
    }
    for (int i = 0; i < 500; i++) {
        inSynInputOutput[i] = 0.000000f;
    }
    for (int i = 0; i < 500; i++) {
        inSynInterOutput[i] = 0.000000f;
    }


}

void initializeAllSparseArrays() {
}

void initSynDelay()
 {
    
}

    void freeMem()
{
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntInput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkInput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VInput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_UInput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntInter));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkInter));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VInter));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_UInter));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntOutput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkOutput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VOutput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_UOutput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynInputInter));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynInputOutput));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynInterOutput));
}

void exitGeNN(){
  freeMem();
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(calcNeurons);
  ret = clReleaseKernel(calcSynapses);
  ret = clReleaseProgram(program);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU()
{
        calcSynapsesCPU(t);
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
