#!/bin/bash
# Part 3 truth + oracle runs, queue A (plain, sigma = 1e-3).
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
run() { nice -n 19 ionice -c3 $PY scratch/p3_truth.py $1 $2 $3 $4 $5 runs/p3/truth_$1_n$2_L$3_s$4_seed$5 $6 >> logs/p3_truth_$1_n$2.log 2>&1; }
for s in 0 1 2 3; do run plain 16 256 0.001 $s 21; done
for s in 2 4 5 7; do run plain 8 256 0.001 $s 21; done
for s in 0 1 2 3; do run plain 16 64 0.001 $s 22; done
for s in 0 2 3 4; do run plain 8 64 0.001 $s 22; done
echo DONE > logs/q_truth_a.done
