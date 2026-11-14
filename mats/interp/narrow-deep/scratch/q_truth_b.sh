#!/bin/bash
# Variance split for all random nets, then Part 3 truth + oracle for the residual nets (coordinator: headline).
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
for sig in 0.001 0; do
  $N9 $PY scratch/p3_varsplit.py plain 16 64 $sig 0,1,2,3 runs/varsplit_plain_n16_L64_s$sig.json >> logs/varsplit.log 2>&1
  $N9 $PY scratch/p3_varsplit.py plain 16 256 $sig 0,1,2,3 runs/varsplit_plain_n16_L256_s$sig.json >> logs/varsplit.log 2>&1
  $N9 $PY scratch/p3_varsplit.py plain 8 64 $sig 0,2,3,4 runs/varsplit_plain_n8_L64_s$sig.json >> logs/varsplit.log 2>&1
  $N9 $PY scratch/p3_varsplit.py plain 8 256 $sig 2,4,5,7 runs/varsplit_plain_n8_L256_s$sig.json >> logs/varsplit.log 2>&1
  for n in 8 16; do for L in 64 256; do
    $N9 $PY scratch/p3_varsplit.py res $n $L $sig 0,1,2,3 runs/varsplit_res_n${n}_L${L}_s$sig.json >> logs/varsplit.log 2>&1
  done; done
done
run() { $N9 $PY scratch/p3_truth.py $1 $2 $3 $4 $5 runs/p3/truth_$1_n$2_L$3_s$4_seed$5 $6 >> logs/p3_truth_$1_n$2.log 2>&1; }
for s in 0 1 2 3; do run res 16 64 0.001 $s 22; done
for s in 0 1 2 3; do run res 8 64 0.001 $s 22; done
for s in 0 1 2 3; do run res 16 256 0.001 $s 21; done
for s in 0 1 2 3; do run res 8 256 0.001 $s 21; done
echo DONE > logs/q_truth_b.done
