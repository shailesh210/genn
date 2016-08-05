

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model MBody_userdef containing general control code.
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
cl_event neuronEvent;
cl_ulong neuronStart, neuronStop;
double neuron_tme;
CStopWatch neuron_timer;
cl_events synapseEvent;
cl_ulong synapseStart, synapseStop;
double synapse_tme;
CStopWatch synapse_timer;
cl_event learningEvent;
cl_ulong learningStart, learningStop;
double learning_tme;
CStopWatch learning_timer;

// ------------------------------------------------------------------------
// neuron variables

cl_mem d_done;
cl_mem d_t;
unsigned int * glbSpkCntPN;
cl_mem d_glbSpkCntPN;
unsigned int * glbSpkPN;
cl_mem d_glbSpkPN;
float * VPN;
cl_mem d_VPN;
uint64_t * seedPN;
cl_mem d_seedPN;
float * spikeTimePN;
cl_mem d_spikeTimePN;
uint64_t * ratesPN;
unsigned int offsetPN;
unsigned int * glbSpkCntKC;
cl_mem d_glbSpkCntKC;
unsigned int * glbSpkKC;
cl_mem d_glbSpkKC;
float * sTKC;
cl_mem d_sTKC;
float * VKC;
cl_mem d_VKC;
float * mKC;
cl_mem d_mKC;
float * hKC;
cl_mem d_hKC;
float * nKC;
cl_mem d_nKC;
unsigned int * glbSpkCntLHI;
cl_mem d_glbSpkCntLHI;
unsigned int * glbSpkLHI;
cl_mem d_glbSpkLHI;
unsigned int * glbSpkCntEvntLHI;
cl_mem d_glbSpkCntEvntLHI;
unsigned int * glbSpkEvntLHI;
cl_mem d_glbSpkEvntLHI;
float * VLHI;
cl_mem d_VLHI;
float * mLHI;
cl_mem d_mLHI;
float * hLHI;
cl_mem d_hLHI;
float * nLHI;
cl_mem d_nLHI;
unsigned int * glbSpkCntDN;
cl_mem d_glbSpkCntDN;
unsigned int * glbSpkDN;
cl_mem d_glbSpkDN;
unsigned int * glbSpkCntEvntDN;
cl_mem d_glbSpkCntEvntDN;
unsigned int * glbSpkEvntDN;
cl_mem d_glbSpkEvntDN;
float * sTDN;
cl_mem d_sTDN;
float * VDN;
cl_mem d_VDN;
float * mDN;
cl_mem d_mDN;
float * hDN;
cl_mem d_hDN;
float * nDN;
cl_mem d_nDN;

// ------------------------------------------------------------------------
// synapse variables

float * inSynPNKC;
cl_mem d_inSynPNKC;
SparseProjection CPNKC;
unsigned int *d_indInGPNKC;
unsigned int *d_indPNKC;
float * gPNKC;
cl_mem d_gPNKC;
float * EEEEPNKC;
cl_mem d_EEEEPNKC;
float * inSynPNLHI;
cl_mem d_inSynPNLHI;
float * gPNLHI;
cl_mem d_gPNLHI;
float * inSynLHIKC;
cl_mem d_inSynLHIKC;
float * inSynKCDN;
cl_mem d_inSynKCDN;
SparseProjection CKCDN;
unsigned int *d_indInGKCDN;
unsigned int *d_indKCDN;
unsigned int *d_revIndInGKCDN;
unsigned int *d_revIndKCDN;
unsigned int *d_remapKCDN;
float * gKCDN;
cl_mem d_gKCDN;
float * gRawKCDN;
cl_mem d_gRawKCDN;
float * inSynDNDN;
cl_mem d_inSynDNDN;

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
void mapBuffer()
{
glbSpkCntPN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntPN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkPN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkPN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
VPN= (float *) clEnqueueMapBuffer(command_queue, d_VPN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
seedPN= (uint64_t *) clEnqueueMapBuffer(command_queue, d_seedPN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(uint64_t), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
spikeTimePN= (float *) clEnqueueMapBuffer(command_queue, d_spikeTimePN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

glbSpkCntKC= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkKC= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
sTKC= (float *) clEnqueueMapBuffer(command_queue, d_sTKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
VKC= (float *) clEnqueueMapBuffer(command_queue, d_VKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
mKC= (float *) clEnqueueMapBuffer(command_queue, d_mKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
hKC= (float *) clEnqueueMapBuffer(command_queue, d_hKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
nKC= (float *) clEnqueueMapBuffer(command_queue, d_nKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

glbSpkCntLHI= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkLHI= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkCntEvntLHI= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntEvntLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkEvntLHI= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkEvntLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
VLHI= (float *) clEnqueueMapBuffer(command_queue, d_VLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
mLHI= (float *) clEnqueueMapBuffer(command_queue, d_mLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
hLHI= (float *) clEnqueueMapBuffer(command_queue, d_hLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
nLHI= (float *) clEnqueueMapBuffer(command_queue, d_nLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

glbSpkCntDN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkDN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkCntEvntDN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntEvntDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
glbSpkEvntDN= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkEvntDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(unsigned int), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
sTDN= (float *) clEnqueueMapBuffer(command_queue, d_sTDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
VDN= (float *) clEnqueueMapBuffer(command_queue, d_VDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
mDN= (float *) clEnqueueMapBuffer(command_queue, d_mDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
hDN= (float *) clEnqueueMapBuffer(command_queue, d_hDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
nDN= (float *) clEnqueueMapBuffer(command_queue, d_nDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

inSynPNKC= (float *) clEnqueueMapBuffer(command_queue, d_inSynPNKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
EEEEPNKC= (float *) clEnqueueMapBuffer(command_queue, d_EEEEPNKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

inSynPNLHI= (float *) clEnqueueMapBuffer(command_queue, d_inSynPNLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 20* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
gPNLHI= (float *) clEnqueueMapBuffer(command_queue, d_gPNLHI, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 2000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

inSynLHIKC= (float *) clEnqueueMapBuffer(command_queue, d_inSynLHIKC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1000* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

inSynKCDN= (float *) clEnqueueMapBuffer(command_queue, d_inSynKCDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

inSynDNDN= (float *) clEnqueueMapBuffer(command_queue, d_inSynDNDN, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 100* sizeof(float), 0, NULL, NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

}
void unmapBuffer()
{
clEnqueueUnmapMemObject(command_queue, d_glbSpkCntPN,glbSpkCntPN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkPN,glbSpkPN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_VPN, VPN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_seedPN, seedPN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_spikeTimePN, spikeTimePN, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_glbSpkCntKC,glbSpkCntKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkKC,glbSpkKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_sTKC,sTKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_VKC, VKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_mKC, mKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_hKC, hKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_nKC, nKC, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_glbSpkCntLHI,glbSpkCntLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkLHI,glbSpkLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkCntEvntLHI,glbSpkCntEvntLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkEvntLHI,glbSpkEvntLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_VLHI, VLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_mLHI, mLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_hLHI, hLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_nLHI, nLHI, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_glbSpkCntDN,glbSpkCntDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkDN,glbSpkDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkCntEvntDN,glbSpkCntEvntDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_glbSpkEvntDN,glbSpkEvntDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_sTDN,sTDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_VDN, VDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_mDN, mDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_hDN, hDN, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_nDN, nDN, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_inSynPNKC, inSynPNKC, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_EEEEPNKC, EEEEPNKC, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_inSynPNLHI, inSynPNLHI, 0, NULL, NULL);
clEnqueueUnmapMemObject(command_queue, d_gPNLHI, gPNLHI, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_inSynLHIKC, inSynLHIKC, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_inSynKCDN, inSynKCDN, 0, NULL, NULL);

clEnqueueUnmapMemObject(command_queue, d_inSynDNDN, inSynDNDN, 0, NULL, NULL);

}
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
learnSynapsesPost = clCreateKernel(program, "learnSynapsesPost", &ret);
CHECK_OPENCL_ERRORS(ret);
    neuron_tme= 0.0;
    synapse_tme= 0.0;
    learning_tme= 0.0;
d_glbSpkCntPN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkPN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VPN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_seedPN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(uint64_t), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_spikeTimePN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_glbSpkCntKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_sTKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_mKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_hKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_nKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_glbSpkCntLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkCntEvntLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkEvntLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_mLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_hLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_nLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_glbSpkCntDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkCntEvntDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_glbSpkEvntDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_sTDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_VDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_mDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_hDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_nDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynPNKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_EEEEPNKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynPNLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,20 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_gPNLHI=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,2000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynLHIKC=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,1000 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynKCDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_inSynDNDN=clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);

d_done =clCreateBuffer(context,  CL_MEM_READ_WRITE ,100 * sizeof(unsigned int), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
d_t =clCreateBuffer(context,  CL_MEM_READ_WRITE ,100 * sizeof(float), NULL, &ret);
CHECK_OPENCL_ERRORS(ret);
CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_t,CL_TRUE, 0, sizeof(float),(void *)&t,0, NULL, NULL));
mapBuffer();
}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) 1234);

    // neuron variables
    glbSpkCntPN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkPN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        VPN[i] = -60.0000f;
    }
    for (int i = 0; i < 100; i++) {
        seedPN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        spikeTimePN[i] = -10.0000f;
    }
    for (int i = 0; i < 100; i++) {
        seedPN[i] = rand();
    }
    glbSpkCntKC[0] = 0;
    for (int i = 0; i < 1000; i++) {
        glbSpkKC[i] = 0;
    }
    for (int i = 0; i < 1000; i++) {
        sTKC[i] = -10.0;
    }
    for (int i = 0; i < 1000; i++) {
        VKC[i] = -60.0000f;
    }
    for (int i = 0; i < 1000; i++) {
        mKC[i] = 0.0529324f;
    }
    for (int i = 0; i < 1000; i++) {
        hKC[i] = 0.317677f;
    }
    for (int i = 0; i < 1000; i++) {
        nKC[i] = 0.596121f;
    }
    glbSpkCntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkLHI[i] = 0;
    }
    glbSpkCntEvntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkEvntLHI[i] = 0;
    }
    for (int i = 0; i < 20; i++) {
        VLHI[i] = -60.0000f;
    }
    for (int i = 0; i < 20; i++) {
        mLHI[i] = 0.0529324f;
    }
    for (int i = 0; i < 20; i++) {
        hLHI[i] = 0.317677f;
    }
    for (int i = 0; i < 20; i++) {
        nLHI[i] = 0.596121f;
    }
    glbSpkCntDN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkDN[i] = 0;
    }
    glbSpkCntEvntDN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkEvntDN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        sTDN[i] = -10.0;
    }
    for (int i = 0; i < 100; i++) {
        VDN[i] = -60.0000f;
    }
    for (int i = 0; i < 100; i++) {
        mDN[i] = 0.0529324f;
    }
    for (int i = 0; i < 100; i++) {
        hDN[i] = 0.317677f;
    }
    for (int i = 0; i < 100; i++) {
        nDN[i] = 0.596121f;
    }

    // synapse variables
    for (int i = 0; i < 1000; i++) {
        inSynPNKC[i] = 0.000000f;
    }
    for (int i = 0; i < 1000; i++) {
        EEEEPNKC[i] = 0.000000f;
    }
    for (int i = 0; i < 20; i++) {
        inSynPNLHI[i] = 0.000000f;
    }
    for (int i = 0; i < 2000; i++) {
        gPNLHI[i] = 0.000000f;
    }
    for (int i = 0; i < 1000; i++) {
        inSynLHIKC[i] = 0.000000f;
    }
    for (int i = 0; i < 100; i++) {
        inSynKCDN[i] = 0.000000f;
    }
    for (int i = 0; i < 100; i++) {
        inSynDNDN[i] = 0.000000f;
    }


unmapBuffer();
set_kernel_arguments();
}

void allocatePNKC(unsigned int connN){
// Allocate host side variables
  CPNKC.connN= connN;
PNKC.indInG = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,101 * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
PNKC.ind = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
  CPNKC.preInd= NULL;
  CPNKC.revIndInG= NULL;
  CPNKC.revInd= NULL;
  CPNKC.remap= NULL;
PNKC.remap = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
// Allocate device side variables
d_indInGPNKC = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (101) ,NULL,&ret);
d_indPNKC = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (CPNKC.connN) ,NULL,&ret);
d_gPNKC = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*(CPNKC.connN) ,NULL,&ret);
}

void createSparseConnectivityFromDensePNKC(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDensePNKC() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocatePNKC(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateKCDN(unsigned int connN){
// Allocate host side variables
  CKCDN.connN= connN;
KCDN.indInG = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,1001 * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
KCDN.ind = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
  CKCDN.preInd= NULL;
KCDN.revIndInG = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,101  * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
KCDN.revInd = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
KCDN.remap = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
KCDN.remap = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
KCDN.remap = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, connN * sizeof(unsigned int),NULL,&ret);
CHECK_OPENCL_ERRORS(ret);
// Allocate device side variables
d_indInGKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (1001) ,NULL,&ret);
d_indKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (CKCDN.connN) ,NULL,&ret);
d_revIndInGKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (101) ,NULL,&ret);
d_revIndKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (CKCDN.connN) ,NULL,&ret);
d_remapKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * (CKCDN.connN) ,NULL,&ret);
d_gKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*(CKCDN.connN) ,NULL,&ret);
d_gRawKCDN = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*(CKCDN.connN) ,NULL,&ret);
}

void createSparseConnectivityFromDenseKCDN(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseKCDN() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateKCDN(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void initializeAllSparseArrays() {
size_t size;
size = CPNKC.connN;
  initializeSparseArray(CPNKC, d_indPNKC, d_indInGPNKC,100);
CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNKC, gPNKC, sizeof(float) * size , cudaMemcpyHostToDevice));
size = CKCDN.connN;
  initializeSparseArray(CKCDN, d_indKCDN, d_indInGKCDN,1000);
  initializeSparseArrayRev(CKCDN,  d_revIndKCDN,  d_revIndInGKCDN,  d_remapKCDN,100);
CHECK_CUDA_ERRORS(cudaMemcpy(d_gKCDN, gKCDN, sizeof(float) * size , cudaMemcpyHostToDevice));
CHECK_CUDA_ERRORS(cudaMemcpy(d_gRawKCDN, gRawKCDN, sizeof(float) * size , cudaMemcpyHostToDevice));	
}

void initMBody_userdef()
 {
    
createPosttoPreArray(1000, 100, &CKCDN);
    initializeAllSparseArrays();
    }

    void freeMem()
{
unmapBuffer();
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntPN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkPN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VPN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_seedPN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_spikeTimePN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_sTKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_mKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_hKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_nKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntEvntLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkEvntLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_mLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_hLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_nLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkCntEvntDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_glbSpkEvntDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_sTDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_VDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_mDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_hDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_nDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynPNKC));
    CPNKC.connN= 0;
CHECK_OPENCL_ERRORS(clReleaseMemObject(CPNKC.indInG));
CHECK_OPENCL_ERRORS(clReleaseMemObject(CPNKC.ind));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_gPNKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_EEEEPNKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynPNLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_gPNLHI));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynLHIKC));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynKCDN));
    CKCDN.connN= 0;
CHECK_OPENCL_ERRORS(clReleaseMemObject(CKCDN.indInG));
CHECK_OPENCL_ERRORS(clReleaseMemObject(CKCDN.ind));
CHECK_OPENCL_ERRORS(clReleaseMemObject(CKCDN.revIndInG));
CHECK_OPENCL_ERRORS(clReleaseMemObject(CKCDN.revInd));
CHECK_OPENCL_ERRORS(clReleaseMemObject(CKCDN.remap));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_gKCDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_gRawKCDN));
CHECK_OPENCL_ERRORS(clReleaseMemObject(d_inSynDNDN));
}

void exitGeNN(){
  freeMem();
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(calcNeurons);
  ret = clReleaseKernel(calcSynapses);
  ret = clReleaseKernel(learnSynapsesPost);
  ret = clReleaseProgram(program);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU()
{
        synapse_timer.startTimer();
        calcSynapsesCPU(t);
        synapse_timer.stopTimer();
        synapse_tme+= synapse_timer.getElapsedTime();
        learning_timer.startTimer();
        learnSynapsesPostHost(t);
        learning_timer.stopTimer();
        learning_tme+= learning_timer.getElapsedTime();
    neuron_timer.startTimer();
    calcNeuronsCPU(t);
    neuron_timer.stopTimer();
    neuron_tme+= neuron_timer.getElapsedTime();
iT++;
t= iT*DT;
}
