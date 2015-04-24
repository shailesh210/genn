#!/bin/bash
set -e #exit if error or segfault. Turn it off for benchmarking -- big networks are expected to fail on the GPU
# bash benchmarkMBody.sh bmtest3 10 250 "what is new" GPUID+2 2>&1 |tee bmout_mbody #fails at 1M KCs on Tesla (hits global mem limit)

CONNPATH=$(pwd);
echo "model path:" $CONNPATH
OUTNAME=$1;
echo "output name:" $OUTNAME
cd MBody1_benchmark_project;
#mkdir -p ~/benchmark/MBody_results

#if [ -d "$BmDir" ]; then
#  echo "benchmarking directory exists. Using input data from" $BmDir 
#  printf "\n"
#else
#  mkdir -p $BmDir
#  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir " and running the test only once (first run for reference)."
#  printf "\n"
#  firstrun=true;
#fi

ntimes=$2
nNeuronsFirst=$3
custommsg=$4
GPUID=$5
FP=DOUBLE

echo "running " ${ntimes} " times starting from " ${nNeuronsFirst}

((nMB=${nNeuronsFirst}/2))

for ((ttest = 1; ttest <= ${ntimes}; ttest++));  
do
  ((nMB=2*${nMB}))
  echo "nMB is " ${nMB}
  printf "\n\n***********************MBody1 GPU generating code ****************************\n"

  if [ -d ${OUTNAME}_output ]; then
    echo ${custommsg} >> ${OUTNAME}_output/${OUTNAME}.time
  else  
    printf "Running for the first time"
  fi


  if [ -d "$GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}" ]; then 
    echo "Dir exists. Copying files and running with the reference input..."
    mkdir -p ${OUTNAME}_output
    cp -R $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/* ${OUTNAME}_output/
    ./generate_run ${GPUID} 100 ${nMB} 20 100 0.0025 ${OUTNAME} MBody1 0 ${FP} 1
  else
    mkdir -p "$GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}"
    echo "Running with new input files."
    ./generate_run ${GPUID} 100 ${nMB} 20 100 0.0025 ${OUTNAME} MBody1 0 ${FP} 0
  fi


  ./generate_run ${GPUID} 100 ${nMB} 20 100 0.0025 ${OUTNAME} MBody1 0 ${FP} 0 
  
  printf "\n #\n # copying \n #\n #\n"
  cp ${OUTNAME}_output/${OUTNAME}.kcdn* $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/ -R
  cp ${OUTNAME}_output/${OUTNAME}.pnkc* $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/ -R
  cp ${OUTNAME}_output/${OUTNAME}.pnlhi* $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/ -R
  cp ${OUTNAME}_output/${OUTNAME}.inpat* $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/ -R

 
  cd ../MBody_userdef_benchmark_project;
 printf "\n\n***********************MBody_userdef GPU generating code ****************************\n"
  #cp -R ../MBody1_project/${OUTNAME}_output .
  mkdir -p ${OUTNAME}_output
  echo ${custommsg} >> ${OUTNAME}_output/${OUTNAME}.time
  printf "With new setup... \n"  >> ${OUTNAME}_output/${OUTNAME}.time
  cp -R $GENN_PATH/userproject/benchmark/MBody_results/${OUTNAME}_output/${nMB}/* ${OUTNAME}_output/
  ./generate_run ${GPUID} 100 ${nMB} 20 100 0.0025 ${OUTNAME} MBody_userdef 0 ${FP} 1

  cd ../MBody1_benchmark_project;


  for dumbcntr in {1..4}
    do
      printf "\n\n***********************MBody1 GPU "${dumbcntr}" nMB = ${nMB} ****************************\n"
      printf "With ref setup... \n"  >> ${OUTNAME}_output/${OUTNAME}.time
      model/classol_sim ${OUTNAME} 1
      printf "\n\n***********************MBody1 CPU "${dumbcntr}" nMB = ${nMB} ****************************\n"
      #model/classol_sim ${OUTNAME} 0	
    
      cd ../MBody_userdef_benchmark_project
      printf "\n\n***********************MBody_userdef GPU "${dumbcntr}" nMB = ${nMB} ****************************\n"
      printf "With ref setup... \n"  >> ${OUTNAME}_output/${OUTNAME}.time
      model/classol_sim ${OUTNAME} 1
      printf "\n\n***********************MBody_userdef CPU "${dumbcntr}"  nMB = ${nMB} ****************************\n"
      #model/classol_sim ${OUTNAME} 0	
      cd ../MBody1_benchmark_project
  done
  echo "ntimes is" ${ntimes} " testt is " ${testt}
done
  cd ..

  #cp MBody1_project ~/benchmark/MBody_results -R
  echo "MBody1 tail"
  tail -n 15 MBody1_benchmark_project/${OUTNAME}_output/${OUTNAME}.time
  echo "MBody_userdef tail"
  tail -n 15 MBody_userdef_benchmark_project/${OUTNAME}_output/${OUTNAME}.time

  echo "Benchmarking complete!"

