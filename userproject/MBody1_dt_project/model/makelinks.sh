#!/bin/bash
if [ $1 == C ];then
	ln -fs classol_sim.cu.create classol_sim.cu
	ln -fs map_classol.cc.create map_classol.cc
	ln -fs MBody1.cc.create MBody1.cc
fi

if [ $1 == R ];then
	ln -fs classol_sim.cu.read classol_sim.cu
	ln -fs map_classol.cc.read map_classol.cc
	ln -fs MBody1.cc.read MBody1.cc
fi

