
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model SynDelay containing the host side code for a GPU (OpenCL) simulator version.
*/
//-------------------------------------------------------------------------


#define ulong unsigned long
// ------------------------------------------------------------------------
// copying things to device

void pushInputStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_VInput, CL_TRUE, 0, 500 * sizeof(float),VInput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_UInput, CL_TRUE, 0, 500 * sizeof(float),UInput,0, NULL, NULL ));
    }

void pushInputSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, 0, 7 * sizeof(unsigned int), glbSpkCntInput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkInput, CL_TRUE, 0, 3500 * sizeof(unsigned int), glbSpkInput,0, NULL, NULL ));
    }

void pushInputSpikeEventsToDevice()
 {
    }

void pushInputCurrentSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, 0, spkQuePtrInput*sizeof(unsigned int), glbSpkCntInput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkInput, CL_TRUE, 0, (glbSpkCntInput[spkQuePtrInput]+(spkQuePtrInput*500)) * sizeof(unsigned int), glbSpkInput,0, NULL, NULL ));
    }

void pushInputCurrentSpikeEventsToDevice()
 {
    }

void pushInterStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_VInter, CL_TRUE, 0, 500 * sizeof(float),VInter,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_UInter, CL_TRUE, 0, 500 * sizeof(float),UInter,0, NULL, NULL ));
    }

void pushInterSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntInter,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkInter, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkInter,0, NULL, NULL ));
    }

void pushInterSpikeEventsToDevice()
 {
    }

void pushInterCurrentSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntInter,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkInter, CL_TRUE, 0, glbSpkCntInter[0] * sizeof(unsigned int), glbSpkInter,0, NULL, NULL ));
    }

void pushInterCurrentSpikeEventsToDevice()
 {
    }

void pushOutputStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_VOutput, CL_TRUE, 0, 500 * sizeof(float),VOutput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_UOutput, CL_TRUE, 0, 500 * sizeof(float),UOutput,0, NULL, NULL ));
    }

void pushOutputSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntOutput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkOutput, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkOutput,0, NULL, NULL ));
    }

void pushOutputSpikeEventsToDevice()
 {
    }

void pushOutputCurrentSpikesToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntOutput,0, NULL, NULL ));
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_glbSpkOutput, CL_TRUE, 0, glbSpkCntOutput[0] * sizeof(unsigned int), glbSpkOutput,0, NULL, NULL ));
    }

void pushOutputCurrentSpikeEventsToDevice()
 {
    }

void pushInputInterStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_inSynInputInter, CL_TRUE, 0, 500 * sizeof(float), inSynInputInter,0, NULL, NULL ));
    }

void pushInputOutputStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_inSynInputOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInputOutput,0, NULL, NULL ));
    }

void pushInterOutputStateToDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueWriteBuffer(command_queue, d_inSynInterOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInterOutput,0, NULL, NULL ));
    }

// ------------------------------------------------------------------------
// copying things from device

void pullInputStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_VInput,CL_TRUE, 0, 500 * sizeof(float),VInput,0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_UInput,CL_TRUE, 0, 500 * sizeof(float),UInput,0, NULL, NULL));
    }

void pullInputSpikeEventsFromDevice()
 {
    }

void pullInputSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInput,CL_TRUE, 0, 7 * sizeof(unsigned int),glbSpkCntInput, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkInput,CL_TRUE, 0, glbSpkCntInput[0] * sizeof(unsigned int),glbSpkInput,0, NULL, NULL));
    }

void pullInputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullInputCurrentSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, 0, spkQuePtrInput*sizeof(unsigned int), glbSpkCntInput, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkInput,CL_TRUE, 0, (glbSpkCntInput[spkQuePtrInput]+(spkQuePtrInput*500)) * sizeof(unsigned int), glbSpkInput,0, NULL, NULL));
    }

void pullInputCurrentSpikeEventsFromDevice()
 {
    }

void pullInterStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_VInter,CL_TRUE, 0, 500 * sizeof(float),VInter,0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_UInter,CL_TRUE, 0, 500 * sizeof(float),UInter,0, NULL, NULL));
    }

void pullInterSpikeEventsFromDevice()
 {
    }

void pullInterSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInter,CL_TRUE, 0, 1 * sizeof(unsigned int),glbSpkCntInter, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkInter,CL_TRUE, 0, glbSpkCntInter[0] * sizeof(unsigned int),glbSpkInter,0, NULL, NULL));
    }

void pullInterSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullInterCurrentSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInter,CL_TRUE, 0, sizeof(unsigned int), glbSpkCntInter, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkInter,CL_TRUE, 0, glbSpkCntInter[0]* sizeof(unsigned int), glbSpkInter, 0, NULL, NULL));
    }

void pullInterCurrentSpikeEventsFromDevice()
 {
    }

void pullOutputStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_VOutput,CL_TRUE, 0, 500 * sizeof(float),VOutput,0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_UOutput,CL_TRUE, 0, 500 * sizeof(float),UOutput,0, NULL, NULL));
    }

void pullOutputSpikeEventsFromDevice()
 {
    }

void pullOutputSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntOutput,CL_TRUE, 0, 1 * sizeof(unsigned int),glbSpkCntOutput, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkOutput,CL_TRUE, 0, glbSpkCntOutput[0] * sizeof(unsigned int),glbSpkOutput,0, NULL, NULL));
    }

void pullOutputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullOutputCurrentSpikesFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntOutput,CL_TRUE, 0, sizeof(unsigned int), glbSpkCntOutput, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkOutput,CL_TRUE, 0, glbSpkCntOutput[0]* sizeof(unsigned int), glbSpkOutput, 0, NULL, NULL));
    }

void pullOutputCurrentSpikeEventsFromDevice()
 {
    }

void pullInputInterStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_inSynInputInter,CL_TRUE, 0, 500 * sizeof(float),inSynInputInter,0, NULL, NULL));
    }

void pullInputOutputStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_inSynInputOutput,CL_TRUE, 0, 500 * sizeof(float),inSynInputOutput,0, NULL, NULL));
    }

void pullInterOutputStateFromDevice()
 {
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_inSynInterOutput,CL_TRUE, 0, 500 * sizeof(float),inSynInterOutput,0, NULL, NULL));
    }

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice()
 {
    pushInputStateToDevice();
    pushInputSpikesToDevice();
    pushInterStateToDevice();
    pushInterSpikesToDevice();
    pushOutputStateToDevice();
    pushOutputSpikesToDevice();
    pushInputInterStateToDevice();
    pushInputOutputStateToDevice();
    pushInterOutputStateToDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushInputSpikesToDevice();
    pushInterSpikesToDevice();
    pushOutputSpikesToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushInputCurrentSpikesToDevice();
    pushInterCurrentSpikesToDevice();
    pushOutputCurrentSpikesToDevice();
    }
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushInputSpikeEventsToDevice();
    pushInterSpikeEventsToDevice();
    pushOutputSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushInputCurrentSpikeEventsToDevice();
    pushInterCurrentSpikeEventsToDevice();
    pushOutputCurrentSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullInputStateFromDevice();
    pullInputSpikesFromDevice();
    pullInterStateFromDevice();
    pullInterSpikesFromDevice();
    pullOutputStateFromDevice();
    pullOutputSpikesFromDevice();
    pullInputInterStateFromDevice();
    pullInputOutputStateFromDevice();
    pullInterOutputStateFromDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
pullInputSpikesFromDevice();
    pullInterSpikesFromDevice();
    pullOutputSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
pullInputCurrentSpikesFromDevice();
    pullInterCurrentSpikesFromDevice();
    pullOutputCurrentSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInput,CL_TRUE, 0, 7* sizeof(unsigned int), glbSpkCntInput, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntInter,CL_TRUE, 0, 1* sizeof(unsigned int), glbSpkCntInter, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clEnqueueReadBuffer(command_queue, d_glbSpkCntOutput,CL_TRUE, 0, 1* sizeof(unsigned int), glbSpkCntOutput, 0, NULL, NULL));
    }

    
// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
pullInputSpikeEventsFromDevice();
    pullInterSpikeEventsFromDevice();
    pullOutputSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
pullInputCurrentSpikeEventsFromDevice();
    pullInterCurrentSpikeEventsFromDevice();
    pullOutputCurrentSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
}

    
// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
//model.padSumSynapseTrgN[model.synapseGrpN - 1] is 1536
    size_t sGlobalSize = 1536;
    size_t sLocalSize = 32;
    
    size_t nLocalSize = 32;
    size_t nGlobalSize = 1536;
    
    // ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 0, sizeof(cl_mem), &d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 1, sizeof(cl_mem), &d_glbSpkInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 2, sizeof(cl_mem), &d_spkQuePtrInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 3, sizeof(cl_mem), &d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 4, sizeof(cl_mem), &d_glbSpkInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 5, sizeof(cl_mem), &d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 6, sizeof(cl_mem), &d_glbSpkOutput));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 7, sizeof(cl_mem), &d_inSynInputInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 8, sizeof(cl_mem), &d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 9, sizeof(cl_mem), &d_inSynInterOutput));
    
    // ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 0, sizeof(cl_mem), &d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 1, sizeof(cl_mem), &d_glbSpkInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 2, sizeof(cl_mem), &d_spkQuePtrInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 3, sizeof(cl_mem), &d_VInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 4, sizeof(cl_mem), &d_UInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 5, sizeof(cl_mem), &d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 6, sizeof(cl_mem), &d_glbSpkInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 7, sizeof(cl_mem), &d_VInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 8, sizeof(cl_mem), &d_UInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 9, sizeof(cl_mem), &d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 10, sizeof(cl_mem), &d_glbSpkOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 11, sizeof(cl_mem), &d_VOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 12, sizeof(cl_mem), &d_UOutput));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 13, sizeof(cl_mem), &d_inSynInputInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 14, sizeof(cl_mem), &d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 15, sizeof(cl_mem), &d_inSynInterOutput));
    
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcSynapses,1, NULL, &sGlobalSize , &sLocalSize, 0, NULL,NULL/* &synapseevent*/));
    spkQuePtrInput = (spkQuePtrInput + 1) % 7;
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcNeurons,1, NULL, &nGlobalSize , &nLocalSize, 0, NULL,NULL/* &learningevent*/));
    iT++;
    t= iT*DT;
    }

    