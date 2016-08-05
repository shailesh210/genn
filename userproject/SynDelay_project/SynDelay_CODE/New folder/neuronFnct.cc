

#ifndef _SynDelay_neuronFnct_cc
#define _SynDelay_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model SynDelay containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group Input
     {
        spkQuePtrInput = (spkQuePtrInput + 1) % 7;
        glbSpkCntInput[spkQuePtrInput] = 0;
        
        for (int n = 0; n < 500; n++) {
            float lV = VInput[n];
            float lU = UInput[n];
            
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkInput[(spkQuePtrInput * 500) + glbSpkCntInput[spkQuePtrInput]++] = n;
                }
            VInput[n] = lV;
            UInput[n] = lU;
            }
        }
    
    // neuron group Inter
     {
        glbSpkCntInter[0] = 0;
        
        for (int n = 0; n < 500; n++) {
            float lV = VInter[n];
            float lU = UInter[n];
            
            float Isyn = 0;
            Isyn += inSynInputInter[n]; inSynInputInter[n]= 0;
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkInter[glbSpkCntInter[0]++] = n;
                }
            VInter[n] = lV;
            UInter[n] = lU;
            // the post-synaptic dynamics
            
            }
        }
    
    // neuron group Output
     {
        glbSpkCntOutput[0] = 0;
        
        for (int n = 0; n < 500; n++) {
            float lV = VOutput[n];
            float lU = UOutput[n];
            
            float Isyn = 0;
            Isyn += inSynInputOutput[n]; inSynInputOutput[n]= 0;
            Isyn += inSynInterOutput[n]; inSynInterOutput[n]= 0;
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkOutput[glbSpkCntOutput[0]++] = n;
                }
            VOutput[n] = lV;
            UOutput[n] = lU;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
            }
        }
    
    }

    #endif
