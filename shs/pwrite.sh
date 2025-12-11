#!/bin/bash
# please run this script with sudo 

./pmem_build.sh

export FRIC_OFFLOAD=1
export FRIC_EXP="PWRITE"
OUTPUT_FILE="./res/pwrite.log"
> $OUTPUT_FILE

NOW=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment start time: $NOW" >> $OUTPUT_FILE

./run.sh $OUTPUT_FILE