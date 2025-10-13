#!/bin/bash

TEST_DIR=$1
BW_TYPE=$2
MODE=$3

WORK_DIR=$(dirname "$(realpath "$0")")

if [ "$BW_TYPE" = "sc" ]; then
    BW_FILE="$TEST_DIR/sc.bw"
elif [ "$BW_TYPE" = "bulk" ]; then
    BW_FILE="$TEST_DIR/bulk.bigwig"
else
    echo "eror:the second parameter must be sc or bulk"
    exit 1
fi

python "$WORK_DIR/database.py" \
  --work_dir "$WORK_DIR" \
  --test_dir "$TEST_DIR/$MODE" \
  --bw_type "$BW_TYPE" \
  --num_workers 64 \
  --motif_family_file "$WORK_DIR/Data/motif_family_9.txt" \
  --bw_filenames "$BW_FILE"