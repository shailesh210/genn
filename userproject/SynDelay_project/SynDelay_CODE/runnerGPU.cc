
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
    VInput= (float *) clEnqueueMapBuffer(command_queue, d_VInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UInput= (float *) clEnqueueMapBuffer(command_queue, d_UInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInputSpikesToDevice()
 {
    glbSpkCntInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 7* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 3500* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInputSpikeEventsToDevice()
 {
    }

void pushInputCurrentSpikesToDevice()
 {
    glbSpkCntInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, spkQuePtrInput[0]* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, (glbSpkCntInput[spkQuePtrInput[0]]+(spkQuePtrInput[0]*500)) * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInputCurrentSpikeEventsToDevice()
 {
    }

void pushInterStateToDevice()
 {
    VInter= (float *) clEnqueueMapBuffer(command_queue, d_VInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UInter= (float *) clEnqueueMapBuffer(command_queue, d_UInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInterSpikesToDevice()
 {
    glbSpkCntInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInterSpikeEventsToDevice()
 {
    }

void pushInterCurrentSpikesToDevice()
 {
    glbSpkCntInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntInter[0]* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInterCurrentSpikeEventsToDevice()
 {
    }

void pushOutputStateToDevice()
 {
    VOutput= (float *) clEnqueueMapBuffer(command_queue, d_VOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UOutput= (float *) clEnqueueMapBuffer(command_queue, d_UOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushOutputSpikesToDevice()
 {
    glbSpkCntOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushOutputSpikeEventsToDevice()
 {
    }

void pushOutputCurrentSpikesToDevice()
 {
    glbSpkCntOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntOutput[0]* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushOutputCurrentSpikeEventsToDevice()
 {
    }

void pushInputInterStateToDevice()
 {
    inSynInputInter= (float *) clEnqueueMapBuffer(command_queue, d_inSynInputInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInputOutputStateToDevice()
 {
    inSynInputOutput= (float *) clEnqueueMapBuffer(command_queue, d_inSynInputOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pushInterOutputStateToDevice()
 {
    inSynInterOutput= (float *) clEnqueueMapBuffer(command_queue, d_inSynInterOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

// ------------------------------------------------------------------------
// unmap things to device

void unmapInputStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VInput, VInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UInput, UInput, 0, NULL, NULL);
    }

void unmapInputSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInput,glbSpkCntInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInput,glbSpkInput, 0, NULL, NULL);
    }

void unmapInputSpikeEventsToDevice()
 {
    }

void unmapInputCurrentSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInput,glbSpkCntInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInput,glbSpkInput, 0, NULL, NULL);
    }

void unmapInputCurrentSpikeEventsToDevice()
 {
    }

void unmapInterStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VInter, VInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UInter, UInter, 0, NULL, NULL);
    }

void unmapInterSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInter,glbSpkCntInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInter,glbSpkInter, 0, NULL, NULL);
    }

void unmapInterSpikeEventsToDevice()
 {
    }

void unmapInterCurrentSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInter,glbSpkCntInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInter,glbSpkInter, 0, NULL, NULL);
    }

void unmapInterCurrentSpikeEventsToDevice()
 {
    }

void unmapOutputStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VOutput, VOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UOutput, UOutput, 0, NULL, NULL);
    }

void unmapOutputSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntOutput,glbSpkCntOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkOutput,glbSpkOutput, 0, NULL, NULL);
    }

void unmapOutputSpikeEventsToDevice()
 {
    }

void unmapOutputCurrentSpikesToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntOutput,glbSpkCntOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkOutput,glbSpkOutput, 0, NULL, NULL);
    }

void unmapOutputCurrentSpikeEventsToDevice()
 {
    }

void unmapInputInterStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInputInter, inSynInputInter, 0, NULL, NULL);
    }

void unmapInputOutputStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInputOutput, inSynInputOutput, 0, NULL, NULL);
    }

void unmapInterOutputStateToDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInterOutput, inSynInterOutput, 0, NULL, NULL);
    }

// ------------------------------------------------------------------------
// copying(map) things from device

void pullInputStateFromDevice()
 {
    VInput= (float*) clEnqueueMapBuffer(command_queue, d_VInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UInput= (float*) clEnqueueMapBuffer(command_queue, d_UInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInputSpikeEventsFromDevice()
 {
    }

void pullInputSpikesFromDevice()
 {
    glbSpkCntInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 7* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntInput[0] * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullInputCurrentSpikesFromDevice()
 {
    glbSpkCntInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, spkQuePtrInput[0]* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, (glbSpkCntInput[spkQuePtrInput[0]]+(spkQuePtrInput[0]*500)) * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInputCurrentSpikeEventsFromDevice()
 {
    }

void pullInterStateFromDevice()
 {
    VInter= (float*) clEnqueueMapBuffer(command_queue, d_VInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UInter= (float*) clEnqueueMapBuffer(command_queue, d_UInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInterSpikeEventsFromDevice()
 {
    }

void pullInterSpikesFromDevice()
 {
    glbSpkCntInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntInter[0] * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInterSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullInterCurrentSpikesFromDevice()
 {
    glbSpkCntInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntInter[0] * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInterCurrentSpikeEventsFromDevice()
 {
    }

void pullOutputStateFromDevice()
 {
    VOutput= (float*) clEnqueueMapBuffer(command_queue, d_VOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    UOutput= (float*) clEnqueueMapBuffer(command_queue, d_UOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullOutputSpikeEventsFromDevice()
 {
    }

void pullOutputSpikesFromDevice()
 {
    glbSpkCntOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1* sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntOutput[0] * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullOutputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullOutputCurrentSpikesFromDevice()
 {
    glbSpkCntOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, glbSpkCntOutput[0] * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullOutputCurrentSpikeEventsFromDevice()
 {
    }

void pullInputInterStateFromDevice()
 {
    inSynInputInter= (float *) clEnqueueMapBuffer(command_queue, d_inSynInputInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInputOutputStateFromDevice()
 {
    inSynInputOutput= (float *) clEnqueueMapBuffer(command_queue, d_inSynInputOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

void pullInterOutputStateFromDevice()
 {
    inSynInterOutput= (float *) clEnqueueMapBuffer(command_queue, d_inSynInterOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 500* sizeof(float), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    }

// ------------------------------------------------------------------------
// unmap things from device

void unmapInputStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VInput, VInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UInput, UInput, 0, NULL, NULL);
    }

void unmapInputSpikeEventsFromDevice()
 {
    }

void unmapInputSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInput,glbSpkCntInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInput,glbSpkInput, 0, NULL, NULL);
    }

void unmapInputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void unmapInputCurrentSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInput,glbSpkCntInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInput,glbSpkInput, 0, NULL, NULL);
    }

void unmapInputCurrentSpikeEventsFromDevice()
 {
    }

void unmapInterStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VInter, VInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UInter, UInter, 0, NULL, NULL);
    }

void unmapInterSpikeEventsFromDevice()
 {
    }

void unmapInterSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInter,glbSpkCntInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInter,glbSpkInter, 0, NULL, NULL);
    }

void unmapInterSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void unmapInterCurrentSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInter,glbSpkCntInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkInter,glbSpkInter, 0, NULL, NULL);
    }

void unmapInterCurrentSpikeEventsFromDevice()
 {
    }

void unmapOutputStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_VOutput, VOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_UOutput, UOutput, 0, NULL, NULL);
    }

void unmapOutputSpikeEventsFromDevice()
 {
    }

void unmapOutputSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntOutput,glbSpkCntOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkOutput,glbSpkOutput, 0, NULL, NULL);
    }

void unmapOutputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void unmapOutputCurrentSpikesFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntOutput,glbSpkCntOutput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkOutput,glbSpkOutput, 0, NULL, NULL);
    }

void unmapOutputCurrentSpikeEventsFromDevice()
 {
    }

void unmapInputInterStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInputInter, inSynInputInter, 0, NULL, NULL);
    }

void unmapInputOutputStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInputOutput, inSynInputOutput, 0, NULL, NULL);
    }

void unmapInterOutputStateFromDevice()
 {
    clEnqueueUnmapMemObject(command_queue, d_inSynInterOutput, inSynInterOutput, 0, NULL, NULL);
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
    
glbSpkCntInput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 7 * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkCntInter= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntInter, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1 * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
    glbSpkCntOutput= (unsigned int *) clEnqueueMapBuffer(command_queue, d_glbSpkCntOutput, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 1 * sizeof(unsigned int), 0, NULL, NULL, &ret);
    CHECK_OPENCL_ERRORS(ret);
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
// global copying values to device
void unmap_copyStateToDevice()
 {
    unmapInputStateToDevice();
    unmapInputSpikesToDevice();
    unmapInterStateToDevice();
    unmapInterSpikesToDevice();
    unmapOutputStateToDevice();
    unmapOutputSpikesToDevice();
    unmapInputInterStateToDevice();
    unmapInputOutputStateToDevice();
    unmapInterOutputStateToDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes to device
void unmap_copySpikesToDevice()
 {
    unmapInputSpikesToDevice();
    unmapInterSpikesToDevice();
    unmapOutputSpikesToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void unmap_copyCurrentSpikesToDevice()
 {
    unmapInputCurrentSpikesToDevice();
    unmapInterCurrentSpikesToDevice();
    unmapOutputCurrentSpikesToDevice();
    }
// ------------------------------------------------------------------------
// global copying spike events to device
void unmap_copySpikeEventsToDevice()
 {
    unmapInputSpikeEventsToDevice();
    unmapInterSpikeEventsToDevice();
    unmapOutputSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void unmap_copyCurrentSpikeEventsToDevice()
 {
    unmapInputCurrentSpikeEventsToDevice();
    unmapInterCurrentSpikeEventsToDevice();
    unmapOutputCurrentSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// global copying values from device
void unmap_copyStateFromDevice()
 {
    unmapInputStateFromDevice();
    unmapInputSpikesFromDevice();
    unmapInterStateFromDevice();
    unmapInterSpikesFromDevice();
    unmapOutputStateFromDevice();
    unmapOutputSpikesFromDevice();
    unmapInputInterStateFromDevice();
    unmapInputOutputStateFromDevice();
    unmapInterOutputStateFromDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes from device
void unmap_copySpikesFromDevice()
 {
    
unmapInputSpikesFromDevice();
    unmapInterSpikesFromDevice();
    unmapOutputSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikes from device
void unmap_copyCurrentSpikesFromDevice()
 {
    
unmapInputCurrentSpikesFromDevice();
    unmapInterCurrentSpikesFromDevice();
    unmapOutputCurrentSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void unmap_copySpikeNFromDevice()
 {
    
clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInput,glbSpkCntInput, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntInter,glbSpkCntInter, 0, NULL, NULL);
    clEnqueueUnmapMemObject(command_queue, d_glbSpkCntOutput,glbSpkCntOutput, 0, NULL, NULL);
    }

    
// ------------------------------------------------------------------------
// global copying spikeEvents from device
void unmap_copySpikeEventsFromDevice()
 {
    
unmapInputSpikeEventsFromDevice();
    unmapInterSpikeEventsFromDevice();
    unmapOutputSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikeEvents from device
void unmap_copyCurrentSpikeEventsFromDevice()
 {
    
unmapInputCurrentSpikeEventsFromDevice();
    unmapInterCurrentSpikeEventsFromDevice();
    unmapOutputCurrentSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void unmap_copySpikeEventNFromDevice()
 {
    
}

    
// ------------------------------------------------------------------------
// setting kernel arguments
void set_kernel_arguments()
 {
    
// ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 0, sizeof(cl_mem), &d_t));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 1, sizeof(cl_mem), &d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 2, sizeof(cl_mem), &d_glbSpkInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 3, sizeof(cl_mem), &d_spkQuePtrInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 4, sizeof(cl_mem), &d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 5, sizeof(cl_mem), &d_glbSpkInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 6, sizeof(cl_mem), &d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 7, sizeof(cl_mem), &d_glbSpkOutput));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 8, sizeof(cl_mem), &d_inSynInputInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 9, sizeof(cl_mem), &d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 10, sizeof(cl_mem), &d_inSynInterOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcSynapses, 11, sizeof(cl_mem), &d_done));
    
    // ------------------------------------------------------------------------
    // neuron variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 0, sizeof(cl_mem), &d_t));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 1, sizeof(cl_mem), &d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 2, sizeof(cl_mem), &d_glbSpkInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 3, sizeof(cl_mem), &d_spkQuePtrInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 4, sizeof(cl_mem), &d_VInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 5, sizeof(cl_mem), &d_UInput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 6, sizeof(cl_mem), &d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 7, sizeof(cl_mem), &d_glbSpkInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 8, sizeof(cl_mem), &d_VInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 9, sizeof(cl_mem), &d_UInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 10, sizeof(cl_mem), &d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 11, sizeof(cl_mem), &d_glbSpkOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 12, sizeof(cl_mem), &d_VOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 13, sizeof(cl_mem), &d_UOutput));
    // ------------------------------------------------------------------------
    // synapse variables
    
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 14, sizeof(cl_mem), &d_inSynInputInter));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 15, sizeof(cl_mem), &d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(clSetKernelArg(calcNeurons, 16, sizeof(cl_mem), &d_inSynInterOutput));
    
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
    
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcSynapses,1, NULL, &sGlobalSize , &sLocalSize, 0, NULL, NULL));
    CHECK_OPENCL_ERRORS(clFinish(command_queue));
    spkQuePtrInput[0] = (spkQuePtrInput[0] + 1) % 7;
    CHECK_OPENCL_ERRORS(clEnqueueNDRangeKernel(command_queue,calcNeurons,1, NULL, &nGlobalSize , &nLocalSize, 0, NULL, 
    NULL ));
    CHECK_OPENCL_ERRORS(clFinish(command_queue));
    iT++;
    t= iT*DT;
    }

    