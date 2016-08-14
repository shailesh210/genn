
#ifndef SPARSEUTILS_CC
#define SPARSEUTILS_CC

#include "sparseUtils.h"
#include "utils.h"
#include <vector>


//---------------------------------------------------------------------
/*! \brief  Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 */
//---------------------------------------------------------------------

void createPosttoPreArray(unsigned int preN, unsigned int postN, SparseProjection * C) {
    vector<vector<unsigned int> > tempvectInd(postN); //temporary vector to keep indices
    vector<vector<unsigned int> > tempvectV(postN); //temporary vector to keep connectivity values
    unsigned int glbcounter = 0;
    
    for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
	for (int j = 0; j < (C->indInG[i+1]-C->indInG[i]); j++){ //for every postsynaptic neuron j
	    tempvectInd[C->ind[C->indInG[i]+j]].push_back(i); //C->ind[C->indInG[i]+j]: index of postsynaptic neuron
	    tempvectV[C->ind[C->indInG[i]+j]].push_back(C->indInG[i]+j); //this should give where we can find the value in the array
	    glbcounter++;
	}
    }
    unsigned int lcounter =0;

    C->revIndInG[0]=0;
    for (int k = 0; k < postN; k++){
	C->revIndInG[k+1]=C->revIndInG[k]+tempvectInd[k].size();
	for (int p = 0; p< tempvectInd[k].size(); p++){ //if k=0?
	    C->revInd[lcounter]=tempvectInd[k][p];
	    C->remap[lcounter]=tempvectV[k][p];
	    lcounter++;
	}
    }
}


//--------------------------------------------------------------------------
/*! \brief Function to create the mapping from the normal index array "ind" to the "reverse" array revInd, i.e. the inverse mapping of remap. 
This is needed if SynapseDynamics accesses pre-synaptic variables.
 */
//--------------------------------------------------------------------------

void createPreIndices(unsigned int preN, unsigned int postN, SparseProjection * C) 
{
    // let's not assume anything and create from the minimum available data, i.e. indInG and ind
    vector<vector<unsigned int> > tempvect(postN); //temporary vector to keep indices
    for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
	for (int j = 0; j < (C->indInG[i+1]-C->indInG[i]); j++){ //for every postsynaptic neuron j
	    C->preInd[C->indInG[i]+j]= i; // simmple array of the presynaptic neuron index of each synapse
	}
    }
}


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Function for initializing conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------
#ifdef OPENCL
	void initializeSparseArray(cl_command_queue *command_queue, SparseProjection C,  cl_mem dInd, cl_mem dIndInG, unsigned int preN)
#else
	void initializeSparseArray(SparseProjection C,  unsigned int * dInd, unsigned int * dIndInG, unsigned int preN)
#endif
{
	#ifdef OPENCL
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dInd, CL_TRUE, 0, C.connN*sizeof(unsigned int), C.ind, 0, NULL, NULL));
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dIndInG, CL_TRUE, 0, (preN+1)*sizeof(unsigned int), C.indInG, 0, NULL, NULL));
		
	#else
		
		CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
	#endif
} 


//--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------
#ifdef OPENCL
	void initializeSparseArrayRev(cl_command_queue *command_queue, SparseProjection C,  cl_mem dRevInd, cl_mem dRevIndInG, cl_mem dRemap, unsigned int postN)
#else
	void initializeSparseArrayRev(SparseProjection C,  unsigned int * dRevInd, unsigned int * dRevIndInG, unsigned int * dRemap, unsigned int postN)
#endif
{
	#ifdef OPENCL
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dRevInd, CL_TRUE, 0, C.connN*sizeof(unsigned int), C.revInd, 0, NULL, NULL));
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dRevIndInG, CL_TRUE, 0, (postN+1)*sizeof(unsigned int), C.revIndInG, 0, NULL, NULL));
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dRemap, CL_TRUE, 0, C.connN*sizeof(unsigned int), C.remap, 0, NULL, NULL));
	#else
		CHECK_CUDA_ERRORS(cudaMemcpy(dRevInd, C.revInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(dRevIndInG, C.revIndInG, (postN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(dRemap, C.remap, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
	#endif
}


//--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance arrays presynaptic indices for sparse matrices on  the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------
#ifdef OPENCL
	void initializeSparseArrayPreInd(cl_command_queue *command_queue, SparseProjection C,  cl_mem dPreInd)
#else
	void initializeSparseArrayPreInd(SparseProjection C,  unsigned int * dPreInd)
#endif
{
	#ifdef OPENCL
		CHECK_CL_ERRORS(clEnqueueWriteBuffer(*command_queue, dPreInd, CL_TRUE, 0, C.connN*sizeof(unsigned int), C.preInd, 0, NULL, NULL));
	#else
		CHECK_CUDA_ERRORS(cudaMemcpy(dPreInd, C.preInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
	#endif
}
#endif

#endif // SPARSEUTILS_CC
