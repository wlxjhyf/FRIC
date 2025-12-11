#!/bin/bash
# please run this script with sudo 

./dax_build.sh

export FRIC_OFFLOAD=1
export FRIC_EXP="MMAP"
OUTPUT_FILE="./res/mmap.log"
> $OUTPUT_FILE

NOW=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment start time: $NOW" >> $OUTPUT_FILE

./run.sh $OUTPUT_FILE