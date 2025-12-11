#!/bin/bash
# please run this script with sudo 

SCRIPT_DIR=$(pwd)
FRIC_DIR="/mnt/data/xujiahao/FRIC"
PYTHON_BIN="/mnt/data/xujiahao/env/torch/bin/python3"
DEVICE="cuda"
OUTPUT_FILE="${1:-./res/default.txt}"



# CTX_LEN 列表
CTX_LIST=(32 128 512 1024)
# CTX_LIST=(32)

for CTX_LEN in "${CTX_LIST[@]}"; do
    cd $FRIC_DIR || exit 1
    echo "Running benchmark with CTX_LEN=$CTX_LEN on $DEVICE "
    OUTPUT=$($PYTHON_BIN -m benchmark.benchmark $DEVICE $CTX_LEN 2>&1)

    TOKENS=$(echo "$OUTPUT" | grep "the prefill tokens' number is" | awk '{print $NF}')
    PERC90=$(echo "$OUTPUT" | grep "90% perc" | awk '{print $(NF-1)}')
    
    cd "$SCRIPT_DIR" || exit 1
    echo "$TOKENS, $PERC90" >> $OUTPUT_FILE
done
