

#ifndef _SynDelay_neuronKrnl_cc
#define _SynDelay_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model SynDelay containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (threadIdx.x == 0) {
        spkCount = 0;
        }
    __syncthreads();
    
    // neuron group Input
    if (id < 512) {
        
        // only do this for existing neurons
        if (id < 500) {
            // pull neuron variables in a coalesced access
            float lV = dd_VInput[id];
            float lU = dd_UInput[id];
            
            float Isyn = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f) {
        lV=(-65.0000f);
        lU+=(6.00000f);
    }
    lV += 0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+(4.00000f)+Isyn)*DT; //at two times for numerical stability
    lV += 0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+(4.00000f)+Isyn)*DT;
    lU += (0.0200000f)*((0.200000f)*lV-lU)*DT;
   //if (lV > 30.0f) { // keep this only for visualisation -- not really necessaary otherwise
   //    lV = 30.0f;
   //}

            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
                }
            dd_VInput[id] = lV;
            dd_UInput[id] = lU;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntInput[dd_spkQuePtrInput], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkInput[(dd_spkQuePtrInput * 500) + posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    // neuron group Inter
    if ((id >= 512) && (id < 1024)) {
        unsigned int lid = id - 512;
        
        // only do this for existing neurons
        if (lid < 500) {
            // pull neuron variables in a coalesced access
            float lV = dd_VInter[lid];
            float lU = dd_UInter[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInputInter = dd_inSynInputInter[lid];
            Isyn += linSynInputInter; linSynInputInter= 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
                if (lV >= 30.0f){
      lV=(-65.0000f);
		  lU+=(6.00000f);
    } 
    lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
    lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
    lU+=(0.0200000f)*((0.200000f)*lV-lU)*DT;
   //if (lV > 30.0f){   //keep this only for visualisation -- not really necessaary otherwise 
	   //  lV=30.0f; 
   //}

            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VInter[lid] = lV;
            dd_UInter[lid] = lU;
            // the post-synaptic dynamics
            
            dd_inSynInputInter[lid] = linSynInputInter;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntInter[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkInter[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    // neuron group Output
    if ((id >= 1024) && (id < 1536)) {
        unsigned int lid = id - 1024;
        
        // only do this for existing neurons
        if (lid < 500) {
            // pull neuron variables in a coalesced access
            float lV = dd_VOutput[lid];
            float lU = dd_UOutput[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInputOutput = dd_inSynInputOutput[lid];
            Isyn += linSynInputOutput; linSynInputOutput= 0;
            // pull inSyn values in a coalesced access
            float linSynInterOutput = dd_inSynInterOutput[lid];
            Isyn += linSynInterOutput; linSynInterOutput= 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
                if (lV >= 30.0f){
      lV=(-65.0000f);
		  lU+=(6.00000f);
    } 
    lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
    lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
    lU+=(0.0200000f)*((0.200000f)*lV-lU)*DT;
   //if (lV > 30.0f){   //keep this only for visualisation -- not really necessaary otherwise 
	   //  lV=30.0f; 
   //}

            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VOutput[lid] = lV;
            dd_UOutput[lid] = lU;
            // the post-synaptic dynamics
            
            dd_inSynInputOutput[lid] = linSynInputOutput;
            
            dd_inSynInterOutput[lid] = linSynInterOutput;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntOutput[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkOutput[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    }

    #endif
