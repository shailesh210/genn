

#ifndef _MBody_individualID_CLsynapseKrnl_cc
#define _MBody_individualID_CLsynapseKrnl_cc
#define BLOCKSZ_SYN 96

//-------------------------------------------------------------------------
/*! \file CLsynapseKrnl.cc

\brief File generated from GeNN for the model MBody_individualID containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

extern "C" __global__ void CLcalcSynapses(float t)
 {
    unsigned int id = BLOCKSZ_SYN * get_group_id(0) + get_local_id(0) ;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __shared__ float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    __shared__ unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    __shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];
    unsigned int lscntEvnt, numSpikeSubsetsEvnt;
    
    // synapse group PNKC
    if (id < 1056) {
        // only do this for existing neurons
        if (id < 1000) {
            linSyn = dd_inSynPNKC[id];
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
                if (id < 1000) {
                    unsigned int gid = (shSpk[j] * 1000 + id);
                    if (B(dd_gpPNKC[gid >> 5], gid & 31)) {
                        ipost = id;
                          addtoinSyn = (0.00250000f);
  linSyn += addtoinSyn;

                        }
                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (id < 1000) {
            dd_inSynPNKC[id] = linSyn;
            }
        }
    
    // synapse group PNLHI
    if ((id >= 1056) && (id < 1152)) {
        unsigned int lid = id - 1056;
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
    if ((id >= 1152) && (id < 2208)) {
        unsigned int lid = id - 1152;
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
    if ((id >= 2208) && (id < 2400)) {
        unsigned int lid = id - 2208;
        // only do this for existing neurons
        if (lid < 100) {
            linSyn = dd_inSynKCDN[lid];
            }
        lscnt = dd_glbSpkCntKC[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (get_local_id(0) < lmax) {
                shSpk[get_local_id(0)] = dd_glbSpkKC[(r * BLOCKSZ_SYN) + get_local_id(0)];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 100) {
                    ipost = lid;
                    addtoinSyn = dd_gKCDN[shSpk[j] * 100+ ipost];
  linSyn += addtoinSyn; 
				  float dt = dd_sTDN[ipost] - t - ((10.0000f)); 
	  float dg = 0;
				  if (dt > (31.2500f))  
				      dg = -((7.50000e-005f)) ; 
			  else if (dt > 0)  
			      dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
  else if (dt > (-25.0125f))  
			      dg = (1.20000e-005f) * dt + ((0.000300000f)); 
  else dg = - ((1.50000e-007f)) ; 
  dd_gRawKCDN[shSpk[j] * 100+ ipost] += dg; 
  dd_gKCDN[shSpk[j] * 100+ ipost]=(0.0150000f)/2 *(tanhf((33.3300f)*(dd_gRawKCDN[shSpk[j] * 100+ ipost] - ((0.00750000f))))+1); 

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 100) {
            dd_inSynKCDN[lid] = linSyn;
            }
        }
    
    // synapse group DNDN
    if ((id >= 2400) && (id < 2592)) {
        unsigned int lid = id - 2400;
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

extern "C" __global__ void CLlearnSynapsesPost(float t)
 {
    unsigned int id = 32 * get_group_id(0)  + get_local_id(0) ;
    __shared__ unsigned int shSpk[32];
    unsigned int lscnt, numSpikeSubsets, lmax, j, r;
    
    // synapse group KCDN
    if (id < 1024) {
        lscnt = dd_glbSpkCntDN[0];
        numSpikeSubsets = (lscnt+31) / 32;
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % 32)+1;
            else lmax = 32;
            if (get_local_id(0)  < lmax) {
                shSpk[get_local_id(0) ] = dd_glbSpkDN[(r * 32) + get_local_id(0) ];
                }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // only work on existing neurons
            if (id < 1000) {
                // loop through all incoming spikes for learning
                for (j = 0; j < lmax; j++) {
                    
                float dt = t - (dd_sTKC[id]) - ((10.0000f)); 
  float dg =0; 
  if (dt > (31.2500f))  
      dg = -((7.50000e-005f)) ; 
   else if (dt > 0)  
      dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
  else if (dt > (-25.0125f))  
      dg = (1.20000e-005f) * dt + ((0.000300000f)); 
  else dg = -((1.50000e-007f)) ; 
  dd_gRawKCDN[id * 100 + shSpk[j]] += dg; 
  dd_gKCDN[id * 100 + shSpk[j]]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(dd_gRawKCDN[id * 100 + shSpk[j]] - ((0.00750000f))))+1); 

                    }
                }
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0)  == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 31) {
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntKC[0] = 0;
                dd_glbSpkCntEvntLHI[0] = 0;
                dd_glbSpkCntLHI[0] = 0;
                dd_glbSpkCntEvntDN[0] = 0;
                dd_glbSpkCntDN[0] = 0;
                d_done = 0;
                }
            }
        }
    }

#endif
