

#ifndef _MBody_individualID_CLneuronKrnl_cc
#define _MBody_individualID_CLneuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file CLneuronKrnl.cc

\brief File generated from GeNN for the model MBody_individualID containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

extern "C" __kernel void calcNeurons(uint64_t * ratesPN, unsigned int offsetPN, float t)
 {
    unsigned int id =  get_global_id(0);
    __shared__ volatile unsigned int posSpkEvnt;
    __shared__ unsigned int shSpkEvnt[64];
    unsigned int spkEvntIdx;
    __shared__ volatile unsigned int spkEvntCount;
    __shared__ unsigned int shSpk[64];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (get_local_id(0) == 0) {
        spkCount = 0;
        }
    if (get_local_id(0) == 1) {
        spkEvntCount = 0;
        }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    // neuron group PN
    if (id < 128) {
        
        // only do this for existing neurons
        if (id < 100) {
            // pull neuron variables in a coalesced access
            float lV = dd_VPN[id];
            uint64_t lseed = dd_seedPN[id];
            float lspikeTime = dd_spikeTimePN[id];
            
            // calculate membrane potential
                uint64_t theRnd;
    if (lV > (-60.0000f)) {
      lV= (-60.0000f);
    }
    else {
      if (t - lspikeTime > ((2.50000f))) {
        MYRAND(lseed,theRnd);
        if (theRnd < *(ratesPN+offsetPN+id)) {
			          lV= (20.0000f);
          lspikeTime= t;
        }
      }
    }

            dd_VPN[id] = lV;
            dd_seedPN[id] = lseed;
            dd_spikeTimePN[id] = lspikeTime;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPN[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkCount) {
            dd_glbSpkPN[posSpk + get_local_id(0) ] = shSpk[get_local_id(0)];
            dd_sTPN[shSpk[get_local_id(0)]] = t;
            }
        }
    
    // neuron group KC
    if ((id >= 128) && (id < 1152)) {
        unsigned int lid = id - 128;
        
        // only do this for existing neurons
        if (lid < 1000) {
            // pull neuron variables in a coalesced access
            float lV = dd_VKC[lid];
            float lm = dd_mKC[lid];
            float lh = dd_hKC[lid];
            float ln = dd_nKC[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNKC = dd_inSynPNKC[lid];
            Isyn += linSynPNKC*((0.000000f)-lV);
            // pull inSyn values in a coalesced access
            float linSynLHIKC = dd_inSynLHIKC[lid];
            Isyn += linSynLHIKC*((-92.0000f)-lV);
            // calculate membrane potential
               float Imem;
    unsigned int mt;
    float mdt= DT/25.0f;
    for (mt=0; mt < 25; mt++) {
      Imem= -(lm*lm*lm*lh*(7.15000f)*(lV-((50.0000f)))+
              ln*ln*ln*ln*(1.43000f)*(lV-((-95.0000f)))+
              (0.0267200f)*(lV-((-63.5630f)))-Isyn);
      float _a;
      if (lV == -52.0f) _a= 1.28f;
      else _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
      float _b;
      if (lV == -25.0f) _b= 1.4f;
      else _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
      lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
      _a= 0.128f*expf((-48.0f-lV)/18.0f);
      _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
      lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
      if (lV == -50.0f) _a= 0.16f;
      else _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
      _b= 0.5f*expf((-55.0f-lV)/40.0f);
      ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
      lV+= Imem/(0.143000f)*mdt;
    }

            dd_VKC[lid] = lV;
            dd_mKC[lid] = lm;
            dd_hKC[lid] = lh;
            dd_nKC[lid] = ln;
            // the post-synaptic dynamics
             	 linSynPNKC*=(0.904837f);

            dd_inSynPNKC[lid] = linSynPNKC;
             	 linSynLHIKC*=(0.935507f);

            dd_inSynLHIKC[lid] = linSynLHIKC;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntKC[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkCount) {
            dd_glbSpkKC[posSpk + get_local_id(0) ] = shSpk[get_local_id(0)];
            dd_sTKC[shSpk[get_local_id(0)]] = t;
            }
        }
    
    // neuron group LHI
    if ((id >= 1152) && (id < 1216)) {
        unsigned int lid = id - 1152;
        
        // only do this for existing neurons
        if (lid < 20) {
            // pull neuron variables in a coalesced access
            float lV = dd_VLHI[lid];
            float lm = dd_mLHI[lid];
            float lh = dd_hLHI[lid];
            float ln = dd_nLHI[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNLHI = dd_inSynPNLHI[lid];
            Isyn += linSynPNLHI*((0.000000f)-lV);
            // calculate membrane potential
               float Imem;
    unsigned int mt;
    float mdt= DT/25.0f;
    for (mt=0; mt < 25; mt++) {
      Imem= -(lm*lm*lm*lh*(7.15000f)*(lV-((50.0000f)))+
              ln*ln*ln*ln*(1.43000f)*(lV-((-95.0000f)))+
              (0.0267200f)*(lV-((-63.5630f)))-Isyn);
      float _a;
      if (lV == -52.0f) _a= 1.28f;
      else _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
      float _b;
      if (lV == -25.0f) _b= 1.4f;
      else _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
      lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
      _a= 0.128f*expf((-48.0f-lV)/18.0f);
      _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
      lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
      if (lV == -50.0f) _a= 0.16f;
      else _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
      _b= 0.5f*expf((-55.0f-lV)/40.0f);
      ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
      lV+= Imem/(0.143000f)*mdt;
    }

            // test for and register a spike-like event
            if ((lV > (-40.0000f))) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
                }
            dd_VLHI[lid] = lV;
            dd_mLHI[lid] = lm;
            dd_hLHI[lid] = lh;
            dd_nLHI[lid] = ln;
            // the post-synaptic dynamics
             	 linSynPNLHI*=(0.904837f);

            dd_inSynPNLHI[lid] = linSynPNLHI;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntLHI[0], spkEvntCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntLHI[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkEvntCount) {
            dd_glbSpkEvntLHI[posSpkEvnt + get_local_id(0) ] = shSpkEvnt[get_local_id(0) ];
            }
        if (get_local_id(0) < spkCount) {
            dd_glbSpkLHI[posSpk + get_local_id(0) ] = shSpk[get_local_id(0)];
            dd_sTLHI[shSpk[get_local_id(0)]] = t;
            }
        }
    
    // neuron group DN
    if ((id >= 1216) && (id < 1344)) {
        unsigned int lid = id - 1216;
        
        // only do this for existing neurons
        if (lid < 100) {
            // pull neuron variables in a coalesced access
            float lV = dd_VDN[lid];
            float lm = dd_mDN[lid];
            float lh = dd_hDN[lid];
            float ln = dd_nDN[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynKCDN = dd_inSynKCDN[lid];
            Isyn += linSynKCDN*((0.000000f)-lV);
            // pull inSyn values in a coalesced access
            float linSynDNDN = dd_inSynDNDN[lid];
            Isyn += linSynDNDN*((-92.0000f)-lV);
            // calculate membrane potential
               float Imem;
    unsigned int mt;
    float mdt= DT/25.0f;
    for (mt=0; mt < 25; mt++) {
      Imem= -(lm*lm*lm*lh*(7.15000f)*(lV-((50.0000f)))+
              ln*ln*ln*ln*(1.43000f)*(lV-((-95.0000f)))+
              (0.0267200f)*(lV-((-63.5630f)))-Isyn);
      float _a;
      if (lV == -52.0f) _a= 1.28f;
      else _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
      float _b;
      if (lV == -25.0f) _b= 1.4f;
      else _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
      lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
      _a= 0.128f*expf((-48.0f-lV)/18.0f);
      _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
      lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
      if (lV == -50.0f) _a= 0.16f;
      else _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
      _b= 0.5f*expf((-55.0f-lV)/40.0f);
      ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
      lV+= Imem/(0.143000f)*mdt;
    }

            // test for and register a spike-like event
            if ((lV > (-30.0000f))) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
                }
            dd_VDN[lid] = lV;
            dd_mDN[lid] = lm;
            dd_hDN[lid] = lh;
            dd_nDN[lid] = ln;
            // the post-synaptic dynamics
             	 linSynKCDN*=(0.980199f);

            dd_inSynKCDN[lid] = linSynKCDN;
             	 linSynDNDN*=(0.960789f);

            dd_inSynDNDN[lid] = linSynDNDN;
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntDN[0], spkEvntCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntDN[0], spkCount);
            }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < spkEvntCount) {
            dd_glbSpkEvntDN[posSpkEvnt + get_local_id(0) ] = shSpkEvnt[get_local_id(0) ];
            }
        if (get_local_id(0) < spkCount) {
            dd_glbSpkDN[posSpk + get_local_id(0) ] = shSpk[get_local_id(0)];
            dd_sTDN[shSpk[get_local_id(0)]] = t;
            }
        }
    
    }

    #endif
