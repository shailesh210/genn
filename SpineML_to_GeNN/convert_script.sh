#!/bin/bash

# BASH SCRIPT TO LINK SPINEML TO GENN
# ALEX COPE - 2013-2014
# UNIVERSITY OF SHEFFIELD

function usage () {
   cat <<EOF

usage: $0 [options]

convert_script_s2g is used to process a SpineML model for GeNN

Options are:

  -w dirpath   Set the working directory - GENN_PATH. If not set defaults
  						 to GENN_PATH sys variable.

  -m modeldir  Set the model directory - the location of the experiment xml
               files.
               This is copied into the output directory and "preflighted" by
               spineml_preflight before the simulation starts.

  -o outdir    Set the output directory for temporary files and data output.

  -e expt      Choose the experiment to run. Each model may have several
               experiments, numbers 0, 1, 2, etc. Here, you can pass the number
               of the experiment to run. E.g. -e 0. Defaults to 0.

  -p "option"  Property change option. This modifies the experiment.xml file to
               add a change to a parameter or the initial value of a state
               variable. The population or projection name must be given, along
               with the parameter/variable property name and the new value.
               These three elements are separated with the colon character. The
               new value can have its dimensions given after the value.

               E.g.: -p "Striatum_D1:tau:45ms" - change param "tau" to 45 ms
               for the population "Striatum_D1".

               Multiple instances of this option may be given. The content of
               each option is passed unmodified to spineml_preflight.

EOF
   exit 0
}

#exit on first error
set -e

MODEL_DIR=$PWD"/model"
LOG_DIR=$PWD"/temp"

# Declare an array for the property change options
declare -a PROPERTY_CHANGES
PCI=0 # Property Change Iterator

# get the command line options...
while getopts i:e:w:srm:o:p:n:a:vV\? opt
do
case "$opt" in
i)  MODEL_DIR="$OPTARG"
;;
e)  EXPERIMENT_NAME="experiment$OPTARG.xml"
;;
w)  GENN_2_BRAHMS_DIR="$OPTARG"
;;
m)  MODEL_DIR="$OPTARG"
;;
o)  OUTPUT_DIR="$OPTARG"
;;
p)  PROPERTY_CHANGES[$PCI]="--property_change=$OPTARG"
PCI=$((PCI+1))
;;
\?) usage
;;
esac
done
shift `expr $OPTIND - 1`

mkdir -p $OUTPUT_DIR/model/
# Move the model into the output dir for processing....
cp $MODEL_DIR/* $OUTPUT_DIR/model/
MODEL_DIR=$OUTPUT_DIR/model/

LOG_DIR=$OUTPUT_DIR/log
mkdir -p $LOG_DIR

# What OS are we?
if [ $(uname) = 'Linux' ]; then
if [ $(uname -i) = 'i686' ]; then
OS='Linux'
else
OS='Linux'
fi
elif [ $(uname) = 'Windows_NT' ] || [ $(uname) = 'MINGW32_NT-6.1' ]; then
OS='Windows'
else
OS='OSX'
fi

# Check we have spineml_preflight
which spineml_preflight >/dev/null
spfrtn=$?
if [ "$spfrtn" -ne 0 ]; then
    echo "Please ensure you installed SpineML_Preflight for the spineml_preflight"
    echo "binary which should be installed somewhere in your \$PATH"
    exit -1
fi

# Preflight the model - note we're outputting the contents of the PROPERTY_CHANGES
# array which contains --property_change=a:b:3 type options.
spineml_preflight -e $MODEL_DIR"/"$EXPERIMENT_NAME "${PROPERTY_CHANGES[@]}"
if [ "$?" -ne "0" ]; then
    echo "Failed to preflight the model." >&2
    exit -1
fi

echo "*Running XSLT" > $MODEL_DIR/time.txt

echo ""
echo "Converting SpineML to GeNN"
echo "Alex Cope             2014"
echo "##########################"
echo ""
echo "Creating extra_neurons.h file with new neuron_body components..."
xsltproc -o extra_neurons.h SpineML_2_GeNN_neurons.xsl $MODEL_DIR/$EXPERIMENT_NAME
echo "Done"
echo "Creating extra_postsynapses.h file with new postsynapse components..."
xsltproc -o extra_postsynapses.h SpineML_2_GeNN_postsynapses.xsl $MODEL_DIR/$EXPERIMENT_NAME
echo "Done"
echo "Creating extra_weightupdates.h file with new weightupdate components..."
xsltproc -o extra_weightupdates.h SpineML_2_GeNN_weightupdates.xsl $MODEL_DIR/$EXPERIMENT_NAME
echo "Done"
echo "Creating model.cc file..."
xsltproc -o model.cc SpineML_2_GeNN_model.xsl $MODEL_DIR/$EXPERIMENT_NAME
echo "Done"
echo "Creating sim.cu file..."
xsltproc --stringparam model_dir "$MODEL_DIR" --stringparam log_dir "$LOG_DIR" -o sim.cu SpineML_2_GeNN_sim.xsl $MODEL_DIR/$EXPERIMENT_NAME
echo "Done"
#exit(0)
echo "Running GeNN code generation..."
if [[ -z ${GENN_PATH+x} ]]; then
echo "Sourcing .bashrc as environment does not seem to be correct"
source ~/.bashrc
fi
if [[ -z ${GENN_PATH+x} ]]; then
error_exit "The system environment is not correctly configured"
fi

#check the directory is there
mkdir -p $GENN_PATH/userproject/model_project
cp extra_neurons.h $GENN_PATH/userproject/model_project/
cp extra_postsynapses.h $GENN_PATH/userproject/model_project/
cp extra_weightupdates.h $GENN_PATH/userproject/model_project/
cp rng.h $GENN_PATH/userproject/model_project/
cp Makefile $GENN_PATH/userproject/model_project/
cp model.cc $GENN_PATH/userproject/model_project/model.cc
cp sim.cu $GENN_PATH/userproject/model_project/sim.cu
if cp $MODEL_DIR/*.bin $GENN_PATH/userproject/model_project/; then
	echo "Copying binary data..."	
fi

echo "*GeNN code-gen" > $MODEL_DIR/time.txt

cd $GENN_PATH/userproject/model_project
../../lib/bin/buildmodel.sh model $DBGMODE

echo "*Compiling..." > $MODEL_DIR/time.txt
make clean
make

#if [ $OS = 'Linux' ]; then
#if mv *.bin bin/linux/release/; then
#	echo "Moving binary data..."	
#fi
#cd bin/linux/release
#fi
#if [ $OS = 'OSX' ]; then
#if mv *.bin bin/darwin/release/; then
#echo "Moving binary data..."
#fi
#cd bin/darwin/release
./sim
#rm *.bin
echo "Done"
echo ""
echo "Finished"
