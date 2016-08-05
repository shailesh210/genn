

#ifndef _MBody_userdef_neuronFnct_cc
#define _MBody_userdef_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model MBody_userdef containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group PN
     {
        glbSpkCntPN[0] = 0;
        
        for (int n = 0; n < 100; n++) {
            float lV = VPN[n];
            uint64_t lseed = seedPN[n];
            float lspikeTime = spikeTimePN[n];
            
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
        if (theRnd < *(ratesPN+offsetPN+n)) {
			          lV= (20.0000f);
          lspikeTime= t;
        }
      }
    }

            // test for and register a true spike
            if ((lV >= (20.0000f)) && !(oldSpike)) {
                glbSpkPN[glbSpkCntPN[0]++] = n;
                }
            VPN[n] = lV;
            seedPN[n] = lseed;
            spikeTimePN[n] = lspikeTime;
            }
        }
    
    // neuron group KC
     {
        glbSpkCntKC[0] = 0;
        
        for (int n = 0; n < 1000; n++) {
            float lV = VKC[n];
            float lm = mKC[n];
            float lh = hKC[n];
            float ln = nKC[n];
            
            float Isyn = 0;
            float lpsEEEEPNKC = EEEEPNKC[n];
            Isyn += inSynPNKC[n]*(lpsEEEEPNKC-lV);
            Isyn += inSynLHIKC[n]*((-92.0000f)-lV);
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
            if ((lV > 0.0f) && !(oldSpike)) {
                glbSpkKC[glbSpkCntKC[0]++] = n;
                sTKC[n] = t;
                }
            VKC[n] = lV;
            mKC[n] = lm;
            hKC[n] = lh;
            nKC[n] = ln;
            // the post-synaptic dynamics
             	 inSynPNKC[n]*=(0.904837f);

            EEEEPNKC[n] = lpsEEEEPNKC;
            // the post-synaptic dynamics
            inSynLHIKC[n]*=(0.935507f);

            }
        }
    
    // neuron group LHI
     {
        glbSpkCntEvntLHI[0] = 0;
        glbSpkCntLHI[0] = 0;
        
        for (int n = 0; n < 20; n++) {
            float lV = VLHI[n];
            float lm = mLHI[n];
            float lh = hLHI[n];
            float ln = nLHI[n];
            
            float Isyn = 0;
            Isyn += inSynPNLHI[n]*((0.000000f)-lV);
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
            if ((    lV > (-40.0000f))) {
                glbSpkEvntLHI[glbSpkCntEvntLHI[0]++] = n;
                }
            // test for and register a true spike
            if ((lV > 0.0f) && !(oldSpike)) {
                glbSpkLHI[glbSpkCntLHI[0]++] = n;
                }
            VLHI[n] = lV;
            mLHI[n] = lm;
            hLHI[n] = lh;
            nLHI[n] = ln;
            // the post-synaptic dynamics
            inSynPNLHI[n]*=(0.904837f);

            }
        }
    
    // neuron group DN
     {
        glbSpkCntEvntDN[0] = 0;
        glbSpkCntDN[0] = 0;
        
        for (int n = 0; n < 100; n++) {
            float lV = VDN[n];
            float lm = mDN[n];
            float lh = hDN[n];
            float ln = nDN[n];
            
            float Isyn = 0;
            Isyn += inSynKCDN[n]*((0.000000f)-lV);
            Isyn += inSynDNDN[n]*((-92.0000f)-lV);
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
            if ((    lV > (-30.0000f))) {
                glbSpkEvntDN[glbSpkCntEvntDN[0]++] = n;
                }
            // test for and register a true spike
            if ((lV > 0.0f) && !(oldSpike)) {
                glbSpkDN[glbSpkCntDN[0]++] = n;
                sTDN[n] = t;
                }
            VDN[n] = lV;
            mDN[n] = lm;
            hDN[n] = lh;
            nDN[n] = ln;
            // the post-synaptic dynamics
            inSynKCDN[n]*=(0.980199f);

            // the post-synaptic dynamics
            inSynDNDN[n]*=(0.960789f);

            }
        }
    
    }

    #endif
