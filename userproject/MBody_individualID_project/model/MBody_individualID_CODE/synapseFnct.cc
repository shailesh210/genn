

#ifndef _MBody_individualID_synapseFnct_cc
#define _MBody_individualID_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model MBody_individualID containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapseDynamicsCPU(float t)
 {
    // execute internal synapse dynamics if any
    }
void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group PNKC
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPN[0]; i++) {
            ipre = glbSpkPN[i];
            for (ipost = 0; ipost < 1000; ipost++) {
                unsigned int gid = (ipre * 100 + ipost);
                if (B(gpPNKC[gid >> 5], gid & 31)) {
                      addtoinSyn = (0.00250000f);
  inSynPNKC[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    // synapse group PNLHI
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPN[0]; i++) {
            ipre = glbSpkPN[i];
            for (ipost = 0; ipost < 20; ipost++) {
                  addtoinSyn = gPNLHI[ipre * 20 + ipost];
  inSynPNLHI[ipost] += addtoinSyn;

                }
            }
        }
    
    // synapse group LHIKC
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntLHI[0]; i++) {
            ipre = glbSpkEvntLHI[i];
            for (ipost = 0; ipost < 1000; ipost++) {
                if (VLHI[ipre] > (-40.0000f)) {
                    addtoinSyn = (0.0500000f) * tanhf((VLHI[ipre] - (-40.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    inSynLHIKC[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    // synapse group KCDN
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntKC[0]; i++) {
            ipre = glbSpkKC[i];
            for (ipost = 0; ipost < 100; ipost++) {
                addtoinSyn = gKCDN[ipre * 100 + ipost];
  inSynKCDN[ipost] += addtoinSyn; 
				  float dt = sTDN[ipost] - t - ((10.0000f)); 
	  float dg = 0;
				  if (dt > (31.2500f))  
				      dg = -((7.50000e-005f)) ; 
			  else if (dt > 0)  
			      dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
  else if (dt > (-25.0125f))  
			      dg = (1.20000e-005f) * dt + ((0.000300000f)); 
  else dg = - ((1.50000e-007f)) ; 
  gRawKCDN[ipre * 100 + ipost] += dg; 
  gKCDN[ipre * 100 + ipost]=(0.0150000f)/2 *(tanhf((33.3300f)*(gRawKCDN[ipre * 100 + ipost] - ((0.00750000f))))+1); 

                }
            }
        }
    
    // synapse group DNDN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntDN[0]; i++) {
            ipre = glbSpkEvntDN[i];
            for (ipost = 0; ipost < 100; ipost++) {
                if (VDN[ipre] > (-30.0000f)) {
                    addtoinSyn = (0.0500000f) * tanhf((VDN[ipre] - (-30.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    inSynDNDN[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    }

void learnSynapsesPostHost(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    unsigned int lSpk;
    
    // synapse group KCDN
     {
        for (ipost = 0; ipost < glbSpkCntDN[0]; ipost++) {
            lSpk = glbSpkDN[ipost];
            for (ipre = 0; ipre < 1000; ipre++) {
                float dt = t - (sTKC[ipre]) - ((10.0000f)); 
  float dg =0; 
  if (dt > (31.2500f))  
      dg = -((7.50000e-005f)) ; 
   else if (dt > 0)  
      dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
  else if (dt > (-25.0125f))  
      dg = (1.20000e-005f) * dt + ((0.000300000f)); 
  else dg = -((1.50000e-007f)) ; 
  gRawKCDN[lSpk + 100 * ipre] += dg; 
  gKCDN[lSpk + 100 * ipre]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(gRawKCDN[lSpk + 100 * ipre] - ((0.00750000f))))+1); 

                }
            }
        }
    }

#endif
