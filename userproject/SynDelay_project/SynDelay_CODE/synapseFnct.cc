

#ifndef _SynDelay_synapseFnct_cc
#define _SynDelay_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model SynDelay containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
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
    
    // synapse group InputInter
     {
        unsigned int delaySlot = (spkQuePtrInput[0] + 4) % 7;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInput[delaySlot]; i++) {
            ipre = glbSpkInput[(delaySlot * 500) + i];
            for (ipost = 0; ipost < 500; ipost++) {
                  addtoinSyn = (0.0600000f);
  inSynInputInter[ipost] += addtoinSyn;

                }
            }
        }
    
    // synapse group InputOutput
     {
        unsigned int delaySlot = (spkQuePtrInput[0] + 1) % 7;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInput[delaySlot]; i++) {
            ipre = glbSpkInput[(delaySlot * 500) + i];
            for (ipost = 0; ipost < 500; ipost++) {
                  addtoinSyn = (0.0300000f);
  inSynInputOutput[ipost] += addtoinSyn;

                }
            }
        }
    
    // synapse group InterOutput
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInter[0]; i++) {
            ipre = glbSpkInter[i];
            for (ipost = 0; ipost < 500; ipost++) {
                  addtoinSyn = (0.0300000f);
  inSynInterOutput[ipost] += addtoinSyn;

                }
            }
        }
    
    }


#endif
