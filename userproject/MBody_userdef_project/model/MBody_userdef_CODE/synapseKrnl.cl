

#ifndef _MBody_userdef_synapseKrnl_cl
#define _MBody_userdef_synapseKrnl_cl
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cl

\brief File generated from GeNN for the model MBody_userdef containing the synapse kernel and learning kernel functions.
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
 __kernel void calcSynapses(__global float *t ,  __global unsigned int* dd_glbSpkCntPN,  __global unsigned int* dd_glbSpkPN,  __global unsigned int* dd_glbSpkCntKC,  __global unsigned int* dd_glbSpkKC, __global float* dd_sTKC,  __global unsigned int* dd_glbSpkCntLHI,  __global unsigned int* dd_glbSpkLHI,  __global unsigned int* dd_glbSpkCntEvntLHI, __global unsigned int* dd_glbSpkEvntLHI,  __global unsigned int* dd_glbSpkCntDN,  __global unsigned int* dd_glbSpkDN,  __global unsigned int* dd_glbSpkCntEvntDN, __global unsigned int* dd_glbSpkEvntDN, __global float* dd_sTDN, __global float* dd_inSynPNKC, __global float* dd_gPNKC, __global float* dd_EEEEPNKC, __global float* dd_inSynPNLHI, __global float* dd_gPNLHI, __global float* dd_inSynLHIKC, __global float* dd_inSynKCDN, __global float* dd_gKCDN, __global float* dd_gRawKCDN, __global float* dd_inSynDNDN, __global unsigned int *d_done )
 {
    unsigned int id = BLOCKSZ_SYN * get_group_id(0) + get_local_id(0);
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __local float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    unsigned int prePos; 
    unsigned int npost; 
    __local unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    __local unsigned int shSpkEvnt[BLOCKSZ_SYN];
    unsigned int lscntEvnt, numSpikeSubsetsEvnt;
    
    // synapse group PNKC
    if (id < 1024) {
        lscnt = dd_glbSpkCntPN[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkPN[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 1000) {
                    prePos = dd_indInGPNKC[shSpk[j]];
                    npost = dd_indInGPNKC[shSpk[j] + 1] - prePos;
                    if (id < npost) {
                        prePos += id;
                        ipost = dd_indPNKC[prePos];
                        addtoinSyn = dd_gPNKC[prePos];
  atomicAddoldGPU(&dd_inSynPNKC[ipost], addtoinSyn);

                        }
                    }
                
                    }
            
                }
        
            
        }
    
    // synapse group PNLHI
    if ((id >= 1024) && (id < 1056)) {
        unsigned int lid = id - 1024;
        // only do this for existing neurons
        if (lid < 20) {
            linSyn = dd_inSynPNLHI[lid];
            }
        lscnt = dd_glbSpkCntPN[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkPN[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 20) {
                    ipost = lid;
                    addtoinSyn = dd_gPNLHI[shSpk[j] * 20+ ipost];
  linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 20) {
            dd_inSynPNLHI[lid] = linSyn;
            }
        }
    
    // synapse group LHIKC
    if ((id >= 1056) && (id < 2080)) {
        unsigned int lid = id - 1056;
        // only do this for existing neurons
        if (lid < 1000) {
            linSyn = dd_inSynLHIKC[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntLHI[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpkEvnt[get_local_id(0)] = dd_glbSpkEvntLHI[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 1000) {
                    ipost = lid;
                    addtoinSyn = (0.0500000f) * tanhf((dd_VLHI[shSpkEvnt[j]] - (-40.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 1000) {
            dd_inSynLHIKC[lid] = linSyn;
            }
        }
    
    // synapse group KCDN
    if ((id >= 2080) && (id < 3104)) {
        unsigned int lid = id - 2080;
        lscnt = dd_glbSpkCntKC[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        if (lid < dd_glbSpkCntKC[0]) {
            int preInd = dd_glbSpkKC[lid];
            prePos = dd_indInGKCDN[preInd];
            npost = dd_indInGKCDN[preInd + 1] - prePos;
            for (int i = 0; i < npost; ++i) {
                	ipost = dd_indKCDN[prePos];
                addtoinSyn = dd_gKCDN[prePos];
		atomicAddoldGPU(&dd_inSynKCDN[ipost], addtoinSyn); 
						float dt = dd_sTDN[ipost] - t - ((10.0000f)); 
		float dg = 0;
		if (dt > (31.2500f))  
		dg = -((7.50000e-005f)); 
		else if (dt > 0.0f)  
		dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
		else if (dt > (-25.0125f))  
		dg = (1.20000e-005f) * dt + ((0.000300000f)); 
		else dg = - ((1.50000e-007f)) ; 
		dd_gRawKCDN[prePos] += dg; 
		dd_gKCDN[prePos]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(dd_gRawKCDN[prePos] - ((0.00750000f))))+1.0f); 

                prePos += 1;
                }
            }
        
        }
    
    // synapse group DNDN
    if ((id >= 3104) && (id < 3232)) {
        unsigned int lid = id - 3104;
        // only do this for existing neurons
        if (lid < 100) {
            linSyn = dd_inSynDNDN[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntDN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpkEvnt[get_local_id(0)] = dd_glbSpkEvntDN[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 100) {
                    ipost = lid;
                    addtoinSyn = (0.0500000f) * tanhf((dd_VDN[shSpkEvnt[j]] - (-30.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 100) {
            dd_inSynDNDN[lid] = linSyn;
            }
        }
    
    }

 __kernel void learnSynapsesPost(__global float *t, __global unsigned int* dd_glbSpkCntPN, __global unsigned int* dd_glbSpkPN, __global float* dd_VPN, __global uint64_t* dd_seedPN, __global float* dd_spikeTimePN, __global uint64_t ** dd_ratesPN, __global unsigned int* dd_offsetPN, __global unsigned int* dd_glbSpkCntKC, __global unsigned int* dd_glbSpkKC, __global float* dd_sTKC, __global float* dd_VKC, __global float* dd_mKC, __global float* dd_hKC, __global float* dd_nKC, __global unsigned int* dd_glbSpkCntLHI, __global unsigned int* dd_glbSpkLHI, __global unsigned int* dd_glbSpkCntEvntLHI, __global unsigned int* dd_glbSpkEvntLHI, __global float* dd_VLHI, __global float* dd_mLHI, __global float* dd_hLHI, __global float* dd_nLHI, __global unsigned int* dd_glbSpkCntDN, __global unsigned int* dd_glbSpkDN, __global unsigned int* dd_glbSpkCntEvntDN, __global unsigned int* dd_glbSpkEvntDN, __global float* dd_sTDN, __global float* dd_VDN, __global float* dd_mDN, __global float* dd_hDN, __global float* dd_nDN, __global float* dd_inSynPNKC, __global float* dd_gPNKC, __global float* dd_EEEEPNKC, __global float* dd_inSynPNLHI, __global float* dd_gPNLHI, __global float* dd_inSynLHIKC, __global float* dd_inSynKCDN, __global float* dd_gKCDN, __global float* dd_gRawKCDN, __global float* dd_inSynDNDN, __global unsigned int *d_done )

 {
    unsigned int id = 32 * get_group_id(0)  + get_local_id(0);
    __local unsigned int shSpk[32];
    unsigned int lscnt, numSpikeSubsets, lmax, j, r;
    
    // synapse group KCDN
    if (id < 128) {
        lscnt = dd_glbSpkCntDN[0];
        numSpikeSubsets = (lscnt+31) / 32;
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % 32)+1;
            else lmax = 32;
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkDN[(r * 32) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // only work on existing neurons
            if (id < 1000) {
                // loop through all incoming spikes for learning
                for (j = 0; j < lmax; j++) {
                    
                unsigned int iprePos = dd_revIndInGKCDN[shSpk[j]];
                    unsigned int npre = dd_revIndInGKCDN[shSpk[j] + 1] - iprePos;
                    if (id < npre) {
                        iprePos += id;
                        float dt = t - (dd_sTKC[dd_revIndKCDN[iprePos]]) - ((10.0000f)); 
		float dg =0; 
		if (dt > (31.2500f))  
		dg = -((7.50000e-005f)) ; 
 		else if (dt > 0.0f)  
		dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
		else if (dt > (-25.0125f))  
		dg = (1.20000e-005f) * dt + ((0.000300000f)); 
		else dg = -((1.50000e-007f)) ; 
		dd_gRawKCDN[dd_remapKCDN[iprePos]] += dg; 
		dd_gKCDN[dd_remapKCDN[iprePos]]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(dd_gRawKCDN[dd_remapKCDN[iprePos]] - ((0.00750000f))))+1.0f);

                        }
                    }
                }
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            j = atomic_add( d_done, 1);
            if (j == 3) {
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntKC[0] = 0;
                dd_glbSpkCntEvntLHI[0] = 0;
                dd_glbSpkCntLHI[0] = 0;
                dd_glbSpkCntEvntDN[0] = 0;
                dd_glbSpkCntDN[0] = 0;
                d_done[0] = 0;
                }
            }
        }
    }

#endif
