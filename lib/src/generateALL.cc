/*--------------------------------------------------------------------------
Author: Thomas Nowotny

Institute: Center for Computational Neuroscience and Robotics
University of Sussex
Falmer, Brighton BN1 9QJ, UK

email to:  T.Nowotny@sussex.ac.uk

initial version: 2010-02-07

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateALL.cc

\brief Main file combining the code for code generation. Part of the code generation section.

The file includes separate files for generating kernels (generateKernels.cc), generating the CPU side code for running simulations on either the CPU or GPU (generateRunner.cc) and for CPU-only simulation code (generateCPU.cc).

*/
//--------------------------------------------------------------------------

#include MODEL
#include "generateALL.h"
#include "generateRunner.h"
#include "generateCPU.h"
#include "generateKernels.h"
#include "global.h"
#include "modelSpec.h"
#include "utils.h"
#include "stringUtils.h"
#include "CodeHelper.h"

#ifndef OPENCL
#include <CL/cl.h>
#include<CL/cl_ext.h>
#endif
#include <cmath>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h> // needed for mkdir
#endif

CodeHelper hlp;
//hlp.setVerbose(true);//this will show the generation of bracketing (brace) levels. Helps to debug a bracketing issue

#ifndef OPENCL
//set device properties for opencl
void get_device_properties(CLDeviceProp *deviceprop, int device_no)
{
	char buffer[10240];
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL));
	deviceprop[device_no].major = (int)(buffer[9] - '0');
	deviceprop[device_no].minor = (int)(buffer[11] - '0');
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &(deviceprop[device_no].MAX_WORK_GROUP_SIZE), NULL));
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &(deviceprop[device_no].DEVICE_LOCAL_MEM_SIZE), NULL));
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &(deviceprop[device_no].DEVICE_MAX_COMPUTE_UNITS), NULL));
	//	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_NAME, sizeof(cl_ulong), ((deviceprop)->DEVICE_NAME), NULL));
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &(deviceprop[device_no].DEVICE_GLOBAL_MEM_SIZE), NULL));
	CHECK_CL_ERRORS(clGetDeviceInfo(device_ids[device_no], CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(cl_uint), &(deviceprop[device_no].REGISTERSS_PER_BLOCK), NULL));
}
/*
// display device properties OPENCL
void show_device_properties(CLDeviceProp *deviceprop, int device_no)
{
cout << deviceprop[device_no].MAX_WORK_GROUP_SIZE <<'\n';							//maxThreadsPerBlock
cout << deviceprop[device_no].DEVICE_LOCAL_MEM_SIZE << '\n';						//sharedMemPerBlock
cout << deviceprop[device_no].DEVICE_MAX_COMPUTE_UNITS << '\n';					//multiProcessorCount
//char DEVICE_NAME[100];									//name
cout << deviceprop[device_no].DEVICE_GLOBAL_MEM_SIZE << '\n';					//totalGlobalMem
cout << deviceprop[device_no].REGISTERSS_PER_BLOCK << '\n';						//regsPerBlock
cout << deviceprop[device_no].MAX_WORK_UNITS_PER_COMPUTE_UNIT << '\n';
}
*/

#endif 	//OPENCL

//--------------------------------------------------------------------------
/*! \brief This function will call the necessary sub-functions to generate the code for simulating a model.
*/
//--------------------------------------------------------------------------

void generate_model_runner(NNmodel &model,  //!< Model description
	string path      //!< Path where the generated code will be deposited
	)
{
#ifdef _WIN32
	_mkdir((path + "\\" + model.name + "_CODE").c_str());
#else // UNIX
	mkdir((path + "/" + model.name + "_CODE").c_str(), 0777);
#endif


	// general shared code for GPU and CPU versions
	genRunner(model, path);

#ifndef CPU_ONLY
	// GPU specific code generation
	genRunnerGPU(model, path);

	// generate neuron kernels
	genNeuronKernel(model, path);

	// generate synapse and learning kernels
	if (model.synapseGrpN > 0) genSynapseKernel(model, path);
#endif

	// Generate the equivalent of neuron kernel
	genNeuronFunction(model, path);

	// Generate the equivalent of synapse and learning kernel
	if (model.synapseGrpN > 0) genSynapseFunction(model, path);

	// Generate the Makefile for the generated code
	genMakefile(model, path);
}


//--------------------------------------------------------------------------
/*!
\brief Helper function that prepares data structures and detects the hardware properties to enable the code generation code that follows.

The main tasks in this function are the detection and characterization of the GPU device present (if any), choosing which GPU device to use, finding and appropriate block size, taking note of the major and minor version of the CUDA enabled device chosen for use, and populating the list of standard neuron models. The chosen device number is returned.
*/
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
void chooseDevice(NNmodel &model, //!< the nn model we are generating code for
	string path     //!< path the generated code will be deposited
	)
{
	const int krnlNo = 4;
	const char *kernelName[krnlNo] = { "calcSynapses", "learnSynapsesPost", "calcSynapseDynamics", "calcNeurons" };
	size_t globalMem, mostGlobalMem = 0;
	int chosenDevice = 0;



	// IF OPTIMISATION IS OFF: Simply choose the device with the most global memory.
	{
		cout << "skipping block size optimisation..." << endl;
		synapseBlkSz = GENN_PREFERENCES::synapseBlockSize;
		learnBlkSz = GENN_PREFERENCES::learningBlockSize;
		synDynBlkSz = GENN_PREFERENCES::synapseDynamicsBlockSize;
		neuronBlkSz = GENN_PREFERENCES::neuronBlockSize;
		if (GENN_PREFERENCES::autoChooseDevice) {
			for (theDevice = 0; theDevice < deviceCount; theDevice++) {

#ifndef OPENCL
				get_device_properties(deviceProp, theDevice);
				//show_device_properties(deviceProp, device);
				globalMem = deviceProp[theDevice].DEVICE_GLOBAL_MEM_SIZE;
#else
				CHECK_CUDA_ERRORS(cudaSetDevice(theDevice));
				CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[theDevice]), theDevice));
				globalMem = deviceProp[theDevice].totalGlobalMem;
#endif //OPENCL
				if (globalMem >= mostGlobalMem) {
					mostGlobalMem = globalMem;
					chosenDevice = theDevice;
				}
			}
			cout << "Using device " << chosenDevice << ", which has " << mostGlobalMem << " bytes of global memory." << endl;
		}
		else {
			chosenDevice = GENN_PREFERENCES::defaultDevice;
		}
	}

	theDevice = chosenDevice;
	model.setPopulationSums();

	ofstream sm_os((path + "/sm_version.mk").c_str());
#ifdef _WIN32
	sm_os << "NVCCFLAGS =$(NVCCFLAGS) -arch sm_";
#else // UNIX
	sm_os << "NVCCFLAGS += -arch sm_";
#endif    //_WIN32
	sm_os << deviceProp[chosenDevice].major << deviceProp[chosenDevice].minor << endl;
	sm_os.close();

	cout << "synapse block size: " << synapseBlkSz << endl;
	cout << "learn block size: " << learnBlkSz << endl;
	cout << "synapseDynamics block size: " << synDynBlkSz << endl;
	cout << "neuron block size: " << neuronBlkSz << endl;
		}
#endif


//--------------------------------------------------------------------------
/*! \brief Main entry point for the generateALL executable that generates
the code for GPU and CPU.

The main function is the entry point for the code generation engine. It
prepares the system and then invokes generate_model_runner to inititate
the different parts of actual code generation.
*/
//--------------------------------------------------------------------------

int main(int argc,     //!< number of arguments; expected to be 2
	char *argv[]  //!< Arguments; expected to contain the target directory for code generation.
	)
{
	if (argc != 2) {
		cerr << "usage: generateALL <target dir>" << endl;
		exit(EXIT_FAILURE);
	}

	cout << "call was ";
	for (int i = 0; i < argc; i++) {
		cout << argv[i] << " ";
	}
	cout << endl;

#ifdef DEBUG
	GENN_PREFERENCES::optimizeCode = false;
	GENN_PREFERENCES::debugCode = true;
#endif // DEBUG

#ifndef CPU_ONLY

#ifndef OPENCL
	CHECK_CL_ERRORS(clGetPlatformIDs(1, &platform_id, &ret_num_platforms));
	CHECK_CL_ERRORS(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, device_ids, (cl_uint *)&deviceCount));
	deviceProp = new CLDeviceProp[deviceCount];
	for (int device = 0; device < deviceCount; device++){
		get_device_properties(deviceProp, device);
	}
#else
	CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
	deviceProp = new cudaDeviceProp[deviceCount];
	for (int device = 0; device < deviceCount; device++) {
		CHECK_CUDA_ERRORS(cudaSetDevice(device));
		CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));
	}
#endif  // OPENCL
#endif // CPU_ONLY

	NNmodel *model = new NNmodel();
#ifdef DT
	model->setDT(DT);
	cout << "Setting integration step size from global DT macro: " << DT << endl;
#endif // DT
	modelDefinition(*model);
	if (!model->final) {
		gennError("Model was not finalized in modelDefinition(). Please call model.finalize().");
	}

	string path = toString(argv[1]);

#ifndef CPU_ONLY
	chooseDevice(*model, path);
#endif // CPU_ONLY
	generate_model_runner(*model, path);

	return EXIT_SUCCESS;
}
