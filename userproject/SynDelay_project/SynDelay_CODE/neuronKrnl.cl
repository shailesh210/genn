

#ifndef _SynDelay_neuronKrnl_cl
#define _SynDelay_neuronKrnl_cl

//-------------------------------------------------------------------------
/*! \file CLneuronKrnl.cl

\brief File generated from GeNN for the model SynDelay containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

#undef DT
#define DT 1.00000f
#ifndef MYRAND
#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);
#endif
#ifndef MYRAND_MAX
#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL
#endif
#define BLOCKSZ_SYN 32
 __kernel void calcNeurons(__global float *t, __global unsigned int* dd_glbSpkCntInput, __global unsigned int* dd_glbSpkInput, __global unsigned int * dd_spkQuePtrInput, __global float* dd_VInput, __global float* dd_UInput, __global unsigned int* dd_glbSpkCntInter, __global unsigned int* dd_glbSpkInter, __global float* dd_VInter, __global float* dd_UInter, __global unsigned int* dd_glbSpkCntOutput, __global unsigned int* dd_glbSpkOutput, __global float* dd_VOutput, __global float* dd_UOutput, __global float* dd_inSynInputInter, __global float* dd_inSynInputOutput, __global float* dd_inSynInterOutput)
 {
    unsigned int id =  get_global_id(0);
    __local unsigned int shSpk[32];
    __local volatile unsigned int posSpk;
    unsigned int spkIdx;
    __local volatile unsigned int spkCount;
    
    if (get_local_id(0) == 0) {
        spkCount = 0;
        }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
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
                spkIdx = atomic_add(&spkCount, 1);
                shSpk[spkIdx] = id;
                }
            dd_VInput[id] = lV;
            dd_UInput[id] = lU;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomic_add( &dd_glbSpkCntInput[*dd_spkQuePtrInput], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkCount) {
            dd_glbSpkInput[(*dd_spkQuePtrInput * 500) + posSpk + get_local_id(0)] = shSpk[get_local_id(0)];
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
                spkIdx = atomic_add(&spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VInter[lid] = lV;
            dd_UInter[lid] = lU;
            // the post-synaptic dynamics	
            
            dd_inSynInputInter[lid] = linSynInputInter;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomic_add( &dd_glbSpkCntInter[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkCount) {
            dd_glbSpkInter[posSpk + get_local_id(0)] = shSpk[get_local_id(0)];
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
                spkIdx = atomic_add(&spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VOutput[lid] = lV;
            dd_UOutput[lid] = lU;
            // the post-synaptic dynamics	
            
            dd_inSynInputOutput[lid] = linSynInputOutput;
            
            dd_inSynInterOutput[lid] = linSynInterOutput;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomic_add( &dd_glbSpkCntOutput[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkCount) {
            dd_glbSpkOutput[posSpk + get_local_id(0)] = shSpk[get_local_id(0)];
            }
        }
    
    }

    #endif
