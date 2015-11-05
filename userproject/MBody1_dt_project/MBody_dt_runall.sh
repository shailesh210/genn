#!/bin/bash
############################################
#
# Time step dependency of the MBody1 example
# by Esin Yavuz 
#
############################################
set -e 
outname=dttest_GPU3

#comment the next two line if you re-run the script (it rewrites the connectivity matrices)
#echo "#define DT 0.1" > $GENN_PATH/userproject/include/timestep.h
#echo "#define PATFTIME 1.5" >> $GENN_PATH/userproject/include/timestep.h 
#echo "#define PAT_TIME 100" >> $GENN_PATH/userproject/include/timestep.h 
#echo "double InputBaseRate= 2e-04;">> $GENN_PATH/userproject/include/timestep.h
#define PAT_TIME 100.0

#./generate_run 1 100 10000 20 100 0.0025 ${outname} MBody1 0 DOUBLE 0

dtarray=(0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.0002)
GPU=1 # set 0 for CPU, 1+GPUID to set GPU
for DTind in {0..9}
do
	echo "#define DT "${dtarray[${DTind}]} > $GENN_PATH/userproject/include/timestep.h 
	#patftime=`echo "1.5*0.1/${dtarray[${DTind}]}"|bc -l`
	#echo "#define PATFTIME "${patftime} >> $GENN_PATH/userproject/include/timestep.h
	#pat_time=`echo "100*0.1/${dtarray[${DTind}]}"|bc -l`
	#echo "#define PAT_TIME "${pat_time} >> $GENN_PATH/userproject/include/timestep.h
	#br=`echo "0.0002*0.1/${dtarray[${DTind}]}"|bc -l`
	#echo "double InputBaseRate="${br}";">> $GENN_PATH/userproject/include/timestep.h
	./generate_run 1 100 1000 20 100 0.0025 ${outname} MBody1 0 DOUBLE 1
	cp ${outname}_output ${outname}_GPU0_dt${dtarray[${DTind}]} -r
	#./generate_run 1 100 1000 20 100 0.0025 ${outname} MBody1 0 DOUBLE 1
	#cp ${outname}_output ${outname}_GPU1_dt${dtarray[${DTind}]} -r
	#./generate_run 0 100 1000 20 100 0.0025 ${outname} MBody1 0 DOUBLE 1
	#cp ${outname}_output ${outname}_CPU0_dt${dtarray[${DTind}]} -r
	#model/classol_sim ${outname} 0
	#cp ${outname}_output ${outname}_CPU3_dt${dtarray[${DTind}]} -r

	#for trial in {1..1}
	#do
		#outdir=${outname}_output"/dttest_DT"${dtarray[${DTind}]}"_"${trial} #when you change DT by hand in MBody1.cc, update it here
		#echo "trial "${trial}
		#model/classol_sim ${outname} 1
		#cp ${outname}_output ${outname}_GPU${trial}_dt${dtarray[${DTind}]} -r
	#done
done
