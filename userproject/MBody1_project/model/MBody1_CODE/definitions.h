

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model MBody1 containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#ifndef DT
#define DT 0.1
#endif
#ifndef scalar
typedef float scalar;
#endif
#ifndef SCALAR_MIN
#define SCALAR_MIN 1.17549e-038f
#endif
#ifndef SCALAR_MAX
#define SCALAR_MAX 3.40282e+038f
#endif
#define glbSpkShiftPN 0
#define glbSpkShiftKC 0
#define glbSpkShiftLHI 0
#define glbSpkShiftDN 0
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN
#define spikeCount_KC glbSpkCntKC[0]
#define spike_KC glbSpkKC
#define spikeCount_LHI glbSpkCntLHI[0]
#define spike_LHI glbSpkLHI
#define spikeEventCount_LHI glbSpkCntEvntLHI[0]
#define spikeEvent_LHI glbSpkEvntLHI
#define spikeCount_DN glbSpkCntDN[0]
#define spike_DN glbSpkDN
#define spikeEventCount_DN glbSpkCntEvntDN[0]
#define spikeEvent_DN glbSpkEvntDN
// ------------------------------------------------------------------------
// copying things to device

void pushPNStateToDevice();
void pushPNSpikesToDevice();
void pushPNSpikeEventsToDevice();
void pushPNCurrentSpikesToDevice();
void pushPNCurrentSpikeEventsToDevice();
void pushKCStateToDevice();
void pushKCSpikesToDevice();
void pushKCSpikeEventsToDevice();
void pushKCCurrentSpikesToDevice();
void pushKCCurrentSpikeEventsToDevice();
void pushLHIStateToDevice();
void pushLHISpikesToDevice();
void pushLHISpikeEventsToDevice();
void pushLHICurrentSpikesToDevice();
void pushLHICurrentSpikeEventsToDevice();
void pushDNStateToDevice();
void pushDNSpikesToDevice();
void pushDNSpikeEventsToDevice();
void pushDNCurrentSpikesToDevice();
void pushDNCurrentSpikeEventsToDevice();
#define pushPNKCToDevice pushPNKCStateToDevice

void pushPNKCStateToDevice();
#define pushPNLHIToDevice pushPNLHIStateToDevice

void pushPNLHIStateToDevice();
#define pushLHIKCToDevice pushLHIKCStateToDevice

void pushLHIKCStateToDevice();
#define pushKCDNToDevice pushKCDNStateToDevice

void pushKCDNStateToDevice();
#define pushDNDNToDevice pushDNDNStateToDevice

void pushDNDNStateToDevice();
// ------------------------------------------------------------------------
// copying things from device

void pullPNStateFromDevice();
void pullPNSpikesFromDevice();
void pullPNSpikeEventsFromDevice();
void pullPNCurrentSpikesFromDevice();
void pullPNCurrentSpikeEventsFromDevice();
void pullKCStateFromDevice();
void pullKCSpikesFromDevice();
void pullKCSpikeEventsFromDevice();
void pullKCCurrentSpikesFromDevice();
void pullKCCurrentSpikeEventsFromDevice();
void pullLHIStateFromDevice();
void pullLHISpikesFromDevice();
void pullLHISpikeEventsFromDevice();
void pullLHICurrentSpikesFromDevice();
void pullLHICurrentSpikeEventsFromDevice();
void pullDNStateFromDevice();
void pullDNSpikesFromDevice();
void pullDNSpikeEventsFromDevice();
void pullDNCurrentSpikesFromDevice();
void pullDNCurrentSpikeEventsFromDevice();
#define pullPNKCFromDevice pullPNKCStateFromDevice

void pullPNKCStateFromDevice();
#define pullPNLHIFromDevice pullPNLHIStateFromDevice

void pullPNLHIStateFromDevice();
#define pullLHIKCFromDevice pullLHIKCStateFromDevice

void pullLHIKCStateFromDevice();
#define pullKCDNFromDevice pullKCDNStateFromDevice

void pullKCDNStateFromDevice();
#define pullDNDNFromDevice pullDNDNStateFromDevice

void pullDNDNStateFromDevice();
// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice();
// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice();
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice();
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice();
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice();
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice();
// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice();
// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice();
// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice();
// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice();
// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice();
// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice();
// ------------------------------------------------------------------------
// the actual time stepping procedure
void stepTimeGPU(unsigned int flags);
// neuron variables
extern unsigned int * glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern float * sTPN;
extern float * VPN;
extern uint64_t * seedPN;
extern float * spikeTimePN;
extern uint64_t * ratesPN;
extern unsigned int offsetPN;
extern unsigned int * glbSpkCntKC;
extern unsigned int * glbSpkKC;
extern float * sTKC;
extern float * VKC;
extern float * mKC;
extern float * hKC;
extern float * nKC;
extern unsigned int * glbSpkCntLHI;
extern unsigned int * glbSpkLHI;
extern unsigned int * glbSpkCntEvntLHI;
extern unsigned int * glbSpkEvntLHI;
extern float * sTLHI;
extern float * VLHI;
extern float * mLHI;
extern float * hLHI;
extern float * nLHI;
extern unsigned int * glbSpkCntDN;
extern unsigned int * glbSpkDN;
extern unsigned int * glbSpkCntEvntDN;
extern unsigned int * glbSpkEvntDN;
extern float * sTDN;
extern float * VDN;
extern float * mDN;
extern float * hDN;
extern float * nDN;

// synapse variables
extern float * inSynPNKC;
extern float * gPNKC;
extern float * inSynPNLHI;
extern float * gPNLHI;
extern float * inSynLHIKC;
extern float * inSynKCDN;
extern float * gKCDN;
extern float * gRawKCDN;
extern float * inSynDNDN;

#endif
