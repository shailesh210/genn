

#ifndef _MBody1_neuronKrnl_cc
#define _MBody1_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model MBody1 containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(uint64_t * ratesPN, unsigned int offsetPN, float t)
 {
    unsigned int id = 64 * blockIdx.x + threadIdx.x;
    __shared__ volatile unsigned int posSpkEvnt;
    __shared__ unsigned int shSpkEvnt[64];
    unsigned int spkEvntIdx;
    __shared__ volatile unsigned int spkEvntCount;
    __shared__ unsigned int shSpk[64];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (threadIdx.x == 0) {
        spkCount = 0;
        }
    if (threadIdx.x == 1) {
        spkEvntCount = 0;
        }
    __syncthreads();
    
    // neuron group PN
    if (id < 128) {
        
        // only do this for existing neurons
        if (id < 100) {
            // pull neuron variables in a coalesced access
            float lV = dd_VPN[id];
            uint64_t lseed = dd_seedPN[id];
            float lspikeTime = dd_spikeTimePN[id];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (20.0000f));
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

            // test for and register a true spike
            if ((lV >= (20.0000f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
                }
            dd_VPN[id] = lV;
            dd_seedPN[id] = lseed;
            dd_spikeTimePN[id] = lspikeTime;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPN[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkPN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTPN[shSpk[threadIdx.x]] = t;
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
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV > 0.0f);
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

            // test for and register a true spike
            if ((lV > 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
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
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntKC[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkKC[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTKC[shSpk[threadIdx.x]] = t;
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
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV > 0.0f);
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
            // test for and register a true spike
            if ((lV > 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VLHI[lid] = lV;
            dd_mLHI[lid] = lm;
            dd_hLHI[lid] = lh;
            dd_nLHI[lid] = ln;
            // the post-synaptic dynamics
             	 linSynPNLHI*=(0.904837f);

            dd_inSynPNLHI[lid] = linSynPNLHI;
            }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntLHI[0], spkEvntCount);
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntLHI[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntLHI[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
            }
        if (threadIdx.x < spkCount) {
            dd_glbSpkLHI[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTLHI[shSpk[threadIdx.x]] = t;
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
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV > 0.0f);
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
            // test for and register a true spike
            if ((lV > 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
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
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntDN[0], spkEvntCount);
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntDN[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntDN[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
            }
        if (threadIdx.x < spkCount) {
            dd_glbSpkDN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTDN[shSpk[threadIdx.x]] = t;
            }
        }
    
    }

    #endif
