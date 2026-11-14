#!/bin/bash
# Part 2 collapse census (PREREG P2, P4). Sequential, niced.
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
S64=$(seq -s, 0 63)
for sig in 0 0.001 0.01; do
  for n in 8 16; do
    nice -n 19 ionice -c3 $PY scratch/p2_collapse.py plain $n 256 $sig $S64 runs/collapse_plain_n${n}_L256_s${sig}.json > logs/collapse_plain_n${n}_L256_s${sig}.log 2>&1
  done
done
for sig in 0 0.001 0.01; do
  for n in 8 16; do
    for L in 64 256; do
      nice -n 19 ionice -c3 $PY scratch/p2_collapse.py res $n $L $sig 0,1,2,3 runs/collapse_res_n${n}_L${L}_s${sig}.json > logs/collapse_res_n${n}_L${L}_s${sig}.log 2>&1
    done
  done
done
echo DONE > logs/q_collapse.done
