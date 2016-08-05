
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model MBody_userdef containing the host side code for a GPU (OpenCL) simulator version.
*/
//-------------------------------------------------------------------------


#define ulong unsigned long
// ------------------------------------------------------------------------
// setting kernel arguments
void set_kernel_arguments()
 {
    
// ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 0, sizeof(cl_mem), &d_t));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 1, sizeof(cl_mem), &d_glbSpkCntPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 2, sizeof(cl_mem), &d_glbSpkPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 3, sizeof(cl_mem), &d_glbSpkCntKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 4, sizeof(cl_mem), &d_glbSpkKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 5, sizeof(cl_mem), &d_sTKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 6, sizeof(cl_mem), &d_glbSpkCntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 7, sizeof(cl_mem), &d_glbSpkLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 8, sizeof(cl_mem), &d_glbSpkCntEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 9, sizeof(cl_mem), &d_glbSpkEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 10, sizeof(cl_mem), &d_glbSpkCntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 11, sizeof(cl_mem), &d_glbSpkDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 12, sizeof(cl_mem), &d_glbSpkCntEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 13, sizeof(cl_mem), &d_glbSpkEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 14, sizeof(cl_mem), &d_sTDN));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 15, sizeof(cl_mem), &d_inSynPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 16, sizeof(cl_mem), &d_gPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 17, sizeof(cl_mem), &d_EEEEPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 18, sizeof(cl_mem), &d_inSynPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 19, sizeof(cl_mem), &d_gPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 20, sizeof(cl_mem), &d_inSynLHIKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 21, sizeof(cl_mem), &d_inSynKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 22, sizeof(cl_mem), &d_gKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 23, sizeof(cl_mem), &d_gRawKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 24, sizeof(cl_mem), &d_inSynDNDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 25, sizeof(cl_mem), &d_done));
    
    // ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 0, sizeof(cl_mem), &d_t));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 1, sizeof(cl_mem), &d_glbSpkCntPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 2, sizeof(cl_mem), &d_glbSpkPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 3, sizeof(cl_mem), &d_glbSpkCntKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 4, sizeof(cl_mem), &d_glbSpkKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 5, sizeof(cl_mem), &d_sTKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 6, sizeof(cl_mem), &d_glbSpkCntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 7, sizeof(cl_mem), &d_glbSpkLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 8, sizeof(cl_mem), &d_glbSpkCntEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 9, sizeof(cl_mem), &d_glbSpkEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 10, sizeof(cl_mem), &d_glbSpkCntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 11, sizeof(cl_mem), &d_glbSpkDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 12, sizeof(cl_mem), &d_glbSpkCntEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 13, sizeof(cl_mem), &d_glbSpkEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 14, sizeof(cl_mem), &d_sTDN));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 15, sizeof(cl_mem), &d_inSynPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 16, sizeof(cl_mem), &d_gPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 17, sizeof(cl_mem), &d_EEEEPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 18, sizeof(cl_mem), &d_inSynPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 19, sizeof(cl_mem), &d_gPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 20, sizeof(cl_mem), &d_inSynLHIKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 21, sizeof(cl_mem), &d_inSynKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 22, sizeof(cl_mem), &d_gKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 23, sizeof(cl_mem), &d_gRawKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 24, sizeof(cl_mem), &d_inSynDNDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(learnSynapsesPost, 25, sizeof(cl_mem), &d_done));
    
    // ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 0, sizeof(cl_mem), &ratesPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 1, sizeof(cl_mem), &offsetPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 2, sizeof(cl_mem), &d_t));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 3, sizeof(cl_mem), &d_glbSpkCntPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 4, sizeof(cl_mem), &d_glbSpkPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 5, sizeof(cl_mem), &d_VPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 6, sizeof(cl_mem), &d_seedPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 7, sizeof(cl_mem), &d_spikeTimePN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 8, sizeof(cl_mem), &d_ratesPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 9, sizeof(cl_mem), &d_offsetPN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 10, sizeof(cl_mem), &d_glbSpkCntKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 11, sizeof(cl_mem), &d_glbSpkKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 12, sizeof(cl_mem), &d_sTKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 13, sizeof(cl_mem), &d_VKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 14, sizeof(cl_mem), &d_mKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 15, sizeof(cl_mem), &d_hKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 16, sizeof(cl_mem), &d_nKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 17, sizeof(cl_mem), &d_glbSpkCntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 18, sizeof(cl_mem), &d_glbSpkLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 19, sizeof(cl_mem), &d_glbSpkCntEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 20, sizeof(cl_mem), &d_glbSpkEvntLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 21, sizeof(cl_mem), &d_VLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 22, sizeof(cl_mem), &d_mLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 23, sizeof(cl_mem), &d_hLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 24, sizeof(cl_mem), &d_nLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 25, sizeof(cl_mem), &d_glbSpkCntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 26, sizeof(cl_mem), &d_glbSpkDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 27, sizeof(cl_mem), &d_glbSpkCntEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 28, sizeof(cl_mem), &d_glbSpkEvntDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 29, sizeof(cl_mem), &d_sTDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 30, sizeof(cl_mem), &d_VDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 31, sizeof(cl_mem), &d_mDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 32, sizeof(cl_mem), &d_hDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 33, sizeof(cl_mem), &d_nDN));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 34, sizeof(cl_mem), &d_inSynPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 35, sizeof(cl_mem), &d_gPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 36, sizeof(cl_mem), &d_EEEEPNKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 37, sizeof(cl_mem), &d_inSynPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 38, sizeof(cl_mem), &d_gPNLHI));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 39, sizeof(cl_mem), &d_inSynLHIKC));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 40, sizeof(cl_mem), &d_inSynKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 41, sizeof(cl_mem), &d_gKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 42, sizeof(cl_mem), &d_gRawKCDN));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 43, sizeof(cl_mem), &d_inSynDNDN));
    
    }

    // ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
//model.padSumSynapseTrgN[model.synapseGrpN - 1] is 3232
    size_t sGlobalSize = 3232;
    size_t sLocalSize = 32;
    
    size_t lGlobalSize = 128;
    size_t lLocalSize = 32;
    
    size_t nLocalSize = 32;
    size_t nGlobalSize = 1312;
    
    cudaEventRecord(synapseStart);
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcSynapses,1, NULL, &sGlobalSize , &sLocalSize, 0, NULL,NULL/* &synapseevent*/));
    CHECK_OPENCL_ERRORS(clFinish(command_queue));
    cudaEventRecord(synapseStop);
    cudaEventRecord(learningStart);
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,learnSynapsesPost,1, NULL, &lGlobalSize , &lLocalSize, 0, NULL,NULL/* &learningevent*/));
    CHECK_OPENCL_ERRORS(clFinish(command_queue));
    cudaEventRecord(learningStop);
    cudaEventRecord(neuronStart);
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcNeurons,1, NULL, &nGlobalSize , &nLocalSize, 0, NULL,NULL/* &learningevent*/));
    CHECK_OPENCL_ERRORS(clFinish(command_queue));
    iT++;
    t= iT*DT;
    }

    