#!/bin/bash
# please run this script with sudo 

export FRIC_OFFLOAD=0
OUTPUT_FILE="./res/pure_inf.log"
> $OUTPUT_FILE

NOW=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment start time: $NOW" >> $OUTPUT_FILE

./run.sh $OUTPUT_FILE