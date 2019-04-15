#!/bin/bash

NUM_LINES=$(cat $1 | wc -l)
NUM_CORES=$(parallel --number-of-cores)

PICS_PER_CORE=$((NUM_LINES/NUM_CORES))

CURRENT_IDX=0
rm download_commands.txt
touch download_commands.txt
for ((i = 1; i <= $NUM_CORES; i++ ))
do
    echo "python download_images.py $1 $CURRENT_IDX $((CURRENT_IDX+PICS_PER_CORE-1)) $2 > core_$i.log" >> download_commands.txt
    CURRENT_IDX=$((CURRENT_IDX+PICS_PER_CORE))
done

parallel < download_commands.txt

