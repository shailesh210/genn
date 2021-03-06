
#ifndef TESTPREVARSINSYNAPSEDYNAMICS_H
#define TESTPREVARSINSYNAPSEDYNAMICS_H

#define DT 0.1f
#define TOTAL_TIME 20.0f
#define REPORT_TIME 1.0f

class preVarsInSynapseDynamics
{

public:
  preVarsInSynapseDynamics();
  ~preVarsInSynapseDynamics();
  void init_synapses();
  void init_neurons();
  void run(float, int);

  float **theW;
};

#endif // TESTPREVARSINSYNAPSEDYNAMICS_H
