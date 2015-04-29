/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file classol_sim.cu

\brief Main entry point for the classol (CLASSification in OLfaction) model simulation. Provided as a part of the complete example of simulating the MBody1 mushroom body model. 
*/
//--------------------------------------------------------------------------


#include "classol_sim.h"

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------


int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: classol_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int which = atoi(argv[2]);
  string OutDir = toString(argv[1]) +"_output";
  string name;
  name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  

  patSetTime= (int) (PAT_TIME/DT);
  patFireTime= (int) (PATFTIME/DT);
  fprintf(stdout, "# DT %f \n", DT);
  fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stdout, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
  fprintf(stdout, "# PATFTIME %f \n", PATFTIME); 
  fprintf(stdout, "# patFireTime %d \n", patFireTime);
  fprintf(stdout, "# PAT_TIME %f \n", PAT_TIME);
  fprintf(stdout, "# patSetTime %d \n", patSetTime);
  fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);
  
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.st"); 
  FILE *osf2= fopen(name.c_str(),"w");

  //-----------------------------------------------------------------
  // build the neuronal circuitery
  classol locust;

  fprintf(stdout, "# reading PN-KC synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnkc");
  FILE *f= fopen(name.c_str(),"rb");
  locust.read_pnkcsyns(f);
  fclose(f);

  fprintf(stdout, "# reading PN-LHI synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnlhi");
  f= fopen(name.c_str(), "rb");
  locust.read_pnlhisyns(f);
  fclose(f);   
  
  fprintf(stdout, "# reading KC-DN synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".kcdn");
  f= fopen(name.c_str(), "rb");
  locust.read_kcdnsyns(f);

  fprintf(stdout, "# reading input patterns ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".inpat");
  f= fopen(name.c_str(), "rb");
  locust.read_input_patterns(f);
  fclose(f);

  locust.generate_baserates();
  if (which == GPU) {
    locust.allocate_device_mem_patterns();
  }
  locust.init(which);         // this includes copying g's for the GPU version

  fprintf(stdout, "# neuronal circuitery built, start computation ... \n\n");

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stdout, "# We are running with fixed time step %f \n", DT);
  fprintf(stdout, "# initial wait time execution ... \n");

  t= 0.0;
  int done= 0;
  float last_t_report=  t;
  timer.startTimer();
  
  if (which == GPU){   
    while (!done) 
    {    
      locust.runGPU(DT); // run next batch
      locust.getSpikeNumbersFromGPU();
      locust.getSpikesFromGPU();
    
      locust.sum_spikes();
      locust.output_spikes(osf2, which);
/*      pullDNStateFromDevice();
      fprintf(osf,"%.9g ", t);
      for (int i_dn = 0; i_dn < locust.model.neuronN[3]; i_dn++){
         fprintf(osf,"%.9g ", VDN[i_dn]);
      }
      fprintf(osf,"\n");
*/
      // report progress
      if (t - last_t_report >= T_REPORT_TME)
      {
        fprintf(stdout, "time %f \n", t);
        last_t_report= t;
      }
      // output synapses occasionally
      done= (t >= TOTAL_TME);
    }
  }
  
  if (which == CPU){   
    while (!done) 
    {
      locust.runCPU(DT); // run next batch
      locust.sum_spikes();
      locust.output_spikes(osf2, which);
  /*    fprintf(osf,"%.9g ", t);
      for (int i_dn = 0; i_dn < locust.model.neuronN[3]; i_dn++){
         fprintf(osf,"%.9g ", VDN[i_dn]);
      }
      fprintf(osf,"\n");
*/
      // report progress
      if (t - last_t_report >= T_REPORT_TME)
      {
        fprintf(stdout, "time %f \n", t);
        last_t_report= t;
      }
      done= (t >= TOTAL_TME);
    }
  }
  timer.stopTimer();
  if (which == GPU) pullDNStateFromDevice();
  cerr << "output files are created under the current directory." << endl;
  fprintf(timef, "%d %u %u %u %u %u %.4f %.2f %.1f %.2f\n",which, locust.model.sumNeuronN[locust.model.neuronGrpN-1], locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);
fprintf(stdout, "GPU=%d, %u neurons, %u PN spikes, %u KC spikes, %u LHI spikes, %u DN spikes, simulation took %.4f secs, VDN[0]=%.2f DT=%.1f %.2f\n",which, locust.model.sumNeuronN[locust.model.neuronGrpN-1], locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);
  fclose(osf);
  fclose(osf2);
  fclose(timef);

  if (which == GPU) {
    locust.free_device_mem();
  }
  
  return 0;
}
