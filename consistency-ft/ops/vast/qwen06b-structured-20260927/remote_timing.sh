#!/bin/bash
# Timing: one epoch on a 16/8/8-pair subset with the shared-prefix path, two lanes on the one GPU.
cd /workspace/consistency
python code/scripts/make_subset.py data/qwen06b-structured data/timing 16,8,8
cd code; export PYTHONPATH=.:scripts
M=../models/Qwen3-0.6B; R=../runs-timing; mkdir -p $R
C="--data-root ../data/timing --model $M --arm consistency --train-labels full --device cuda:0 --epochs 1 --batch-size 8 --eval-batch-size 8 --max-length 4096 --shared-prefix --select-on full --log-every 1"
t() { name=$1; shift; python scripts/train_supervised_pairs.py $C --output $R/$name "$@" > $R/$name.log 2>&1; echo "$name EXIT=$?" >> $R/status.txt; }
( nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > $R/gpu-mem.log ) & MON=$!
t solo; t dual_a & t dual_b & wait
kill $MON; echo TIMING_DONE >> $R/status.txt
