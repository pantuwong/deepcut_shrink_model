#!/bin/bash

END=12
for i in $(seq 6 $END)
do 
    echo "Trying to zero conv $i"
    DEEPCUT_ZERO_LAYER="conv-$i" python tokenise.py manual --model "zero-conv-$i"
done