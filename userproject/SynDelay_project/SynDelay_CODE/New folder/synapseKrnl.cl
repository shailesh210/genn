

#ifndef _SynDelay_synapseKrnl_cl
#define _SynDelay_synapseKrnl_cl
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file CLsynapseKrnl.cl

\brief File generated from GeNN for the model SynDelay containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

 __kernel void calcSynapses(float t ,  unsigned int* dd_glbSpkCntInput,  unsigned int* dd_glbSpkInput, unsigned int * dd_spkQuePtrInput,  unsigned int* dd_glbSpkCntInter,  unsigned int* dd_glbSpkInter,  unsigned int* dd_glbSpkCntOutput,  unsigned int* dd_glbSpkOutput, float* dd_inSynInputInter, float* dd_inSynInputOutput, float* dd_inSynInterOutput)
 {
    unsigned int id = BLOCKSZ_SYN * get_group_id(0) + get_local_id(0) ;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __local float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    __local unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    
    // synapse group InputInter
    if (id < 512) {
        unsigned int delaySlot = (*dd_spkQuePtrInput + 4) % 7;
        // only do this for existing neurons
        if (id < 500) {
            linSyn = dd_inSynInputInter[id];
            }
        lscnt = dd_glbSpkCntInput[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkInput[(delaySlot * 500) + (r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 500) {
                    ipost = id;
                      addtoinSyn = (0.0600000f);
  linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (id < 500) {
            dd_inSynInputInter[id] = linSyn;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0)  == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 47) {
                *dd_spkQuePtrInput = (*dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[*dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
                }
            }
        }
    
    // synapse group InputOutput
    if ((id >= 512) && (id < 1024)) {
        unsigned int lid = id - 512;
        unsigned int delaySlot = (*dd_spkQuePtrInput + 1) % 7;
        // only do this for existing neurons
        if (lid < 500) {
            linSyn = dd_inSynInputOutput[lid];
            }
        lscnt = dd_glbSpkCntInput[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkInput[(delaySlot * 500) + (r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 500) {
                    ipost = lid;
                      addtoinSyn = (0.0300000f);
  linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 500) {
            dd_inSynInputOutput[lid] = linSyn;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0)  == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 47) {
                *dd_spkQuePtrInput = (*dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[*dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
                }
            }
        }
    
    // synapse group InterOutput
    if ((id >= 1024) && (id < 1536)) {
        unsigned int lid = id - 1024;
        // only do this for existing neurons
        if (lid < 500) {
            linSyn = dd_inSynInterOutput[lid];
            }
        lscnt = dd_glbSpkCntInter[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkInter[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 500) {
                    ipost = lid;
                      addtoinSyn = (0.0300000f);
  linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 500) {
            dd_inSynInterOutput[lid] = linSyn;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0)  == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 47) {
                *dd_spkQuePtrInput = (*dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[*dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
                }
            }
        }
    
    }


#endif
