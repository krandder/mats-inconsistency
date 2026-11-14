#!/bin/bash
# Timing on a 40-pair train / 20 val / 20 test subset of the disjoint data, one epoch.
cd /workspace/consistency
python code/scripts/make_subset.py data/jev-disjoint-v1-filtered-20260919 data/timing 40,20,20
cd code; export PYTHONPATH=.:scripts
M=../models/Qwen3-0.6B; R=../runs-timing; mkdir -p $R
C="--data-root ../data/timing --model $M --arm consistency --train-labels one --device cuda:0 --epochs 1 --batch-size 8 --eval-batch-size 8 --max-length 1280 --log-every 1"
t() { name=$1; gpu=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python scripts/train_supervised_pairs.py $C --output $R/$name "$@" > $R/$name.log 2>&1; echo "$name EXIT=$?" >> $R/status.txt; }
NG=$(nvidia-smi -L | wc -l)
( nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > $R/gpu-mem.log ) & MON=$!
if [ "$NG" -ge 4 ]; then
  t ppf1 0 --pairs-per-forward 1 &
  t ppf4gc 1 --pairs-per-forward 4 --grad-checkpointing &
  t ppf8gc 2 --pairs-per-forward 8 --grad-checkpointing &
  t dual8gc_a 3 --pairs-per-forward 8 --grad-checkpointing &
  t dual8gc_b 3 --pairs-per-forward 8 --grad-checkpointing &
else
  t ppf8gc 0 --pairs-per-forward 8 --grad-checkpointing &
  t dual8gc_a 1 --pairs-per-forward 8 --grad-checkpointing &
  t dual8gc_b 1 --pairs-per-forward 8 --grad-checkpointing &
fi
N=$([ "$NG" -ge 4 ] && echo 5 || echo 3)
until [ "$(grep -c EXIT $R/status.txt 2>/dev/null)" -ge "$N" ]; do sleep 5; done
kill $MON; echo TIMING_DONE >> $R/status.txt
