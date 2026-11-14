#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
for sig in 0.001 0; do
  for n in 16 8; do
    for L in 64 256; do
      nice -n 19 ionice -c3 $PY scratch/p2_jac.py res $n $L $sig 0,1,2,3 runs/jac_res_n${n}_L${L}_s${sig}.json > logs/jac_res_n${n}_L${L}_s${sig}.log 2>&1
    done
  done
done
echo DONE > logs/q_jac_b.done
