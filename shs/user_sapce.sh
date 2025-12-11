#!/bin/bash
# please run this script with sudo 

export FRIC_OFFLOAD=1
export FRIC_EXP="USER_SPACE"
OUTPUT_FILE="./res/user_space.log"
> $OUTPUT_FILE

NOW=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment start time: $NOW" >> $OUTPUT_FILE

./run.sh $OUTPUT_FILE