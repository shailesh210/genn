//--------------------------------------------------------------------------
//   Author:    James Turner
//  
//   Institute: Center for Computational Neuroscience and Robotics
//              University of Sussex
//              Falmer, Brighton BN1 9QJ, UK 
//  
//   email to:  J.P.Turner@sussex.ac.uk
//  
//--------------------------------------------------------------------------

#ifndef SYNDELAYSIM_CU
#define SYNDELAYSIM_CU

#include <cstdlib>
#include <iostream>
#include <fstream>
#include<conio.h>

using namespace std;

#include "hr_time.h"
#include "utils.h"
#include "stringUtils.h"

#include "SynDelaySim.h"
#include "SynDelay_CODE/definitions.h"


SynDelay::SynDelay(bool usingGPU)
{
  this->usingGPU = usingGPU;
  allocateMem();	
  #ifndef OPENCL 
	copyStateToDevice();
  #endif 
  initialize();
  #ifndef OPENCL 
	unmap_copyStateToDevice();
  #endif 
}

SynDelay::~SynDelay()
{
  freeMem();
}

void SynDelay::run(float t)
{
  if (usingGPU)
  {
#ifndef CPU_ONLY
          
    stepTimeGPU();
	#ifndef OPENCL
		copyStateFromDevice();
	#else
		copyStateFromDevice();
	#endif
#endif // CPU_ONLY
  }
  else
  {
    stepTimeCPU();
  }
}


/*====================================================================
--------------------------- MAIN FUNCTION ----------------------------
====================================================================*/

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cerr << "usage: SynDelaySim <GPU = 1, CPU = 0> <output label>" << endl;
    return EXIT_FAILURE;
  }

#ifdef CPU_ONLY
  if (atoi(argv[1]) == 1)
  {
    cerr << "Cannot use GPU in a CPU_ONLY binary." << endl;
    cerr << "Recompile without CPU_ONLY to run a GPU simulation." << endl;	
    return EXIT_FAILURE;
  }
#endif // CPU_ONLY  


  SynDelay *sim = new SynDelay(atoi(argv[1]));
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream fileTime;
  ofstream fileV;
  ofstream fileStInput;
  ofstream fileStInter;
  ofstream fileStOutput;
  fileTime.open((outLabel + "_time").c_str(), ios::out | ios::app);
  fileV.open((outLabel + "_Vm").c_str(), ios::out | ios::trunc);
  fileStInput.open((outLabel + "_input_st").c_str(), ios::out | ios::trunc);
  fileStInter.open((outLabel + "_inter_st").c_str(), ios::out | ios::trunc);
  fileStOutput.open((outLabel + "_output_st").c_str(), ios::out | ios::trunc);
  cout << "# DT " << DT << endl;
  cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
  cout << "# REPORT_TIME " << REPORT_TIME << endl;
  cout << "# begin simulating on " << (atoi(argv[1]) ? "GPU" : "CPU") << endl;
  timer->startTimer();
  
  set_kernel_arguments();
  
  for (int j = 0; j < (TOTAL_TIME / DT); j++)
//   for (int j = 0; j < 10; j++)
   {
	//  _getch();
	//  cout<<j<<'\n';
    sim->run(t);
	
    t += DT;
	
	
    fileV << t
	  << " " << VInput[0]
	  << " " << VInter[0]
	  << " " << VOutput[0]
	  << endl;
//cout<<"mickey1\n";
//	cout<<"spkQuePtrInput[0] = " << spkQuePtrInput[0]<<'\n';
//	cout << "glbSpkCntInput[spkQuePtrInput[0]]=" << glbSpkCntInput[spkQuePtrInput[0]] <<'\n';
//	cout << "glbSpkShiftInput= " <<  glbSpkShiftInput <<'\n';
//	cout<< " var = " << var[0]<<'\n';
    for (int i= 0; i < glbSpkCntInput[0]; i++) {
		
//	cout << "glbSpkInput[glbSpkShiftInput]= " <<  glbSpkInput[glbSpkShiftInput+i]<<'\n';
	fileStInput << t << " " << glbSpkInput[i] << endl;
    }
//cout<<"mickey2\n";

//	for (int i= 0; i < 500; i++) {
//	fileStInput << t << " " << VInput[i] << endl;
//    }
 
 
    for (int i= 0; i < glbSpkCntInter[0]; i++) {
	fileStInter << t << " " << glbSpkInter[i] << endl;
    }
//cout<<"mickey3\n";
    for (int i= 0; i < glbSpkCntOutput[0]; i++) {
	fileStOutput << t << " " << glbSpkOutput[i] << endl;
    }

//cout<<"mickey4\n";
    if ((int) t % (int) REPORT_TIME == 0)
    {
      cout << "time " << t << endl;
    }
#ifndef OPENCL
	unmap_copyStateFromDevice();	
#endif
  }
  timer->stopTimer();
  cout << "# done in " << timer->getElapsedTime() << " seconds" << endl;
  fileTime << timer->getElapsedTime() << endl;
  fileTime.close();
  fileV.close();

 
  delete sim;
  delete timer;
  return EXIT_SUCCESS;
}

#endif // SYNDELAYSIM_CU
