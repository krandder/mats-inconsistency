#!/bin/bash
# comparator runs (ARC kprop K=2, K=3f) on eval seed 65, widths 64 and 128
cd /home/agent/workspaces/mats/work/sparse-codes
for n in 64 128; do for s in 0.01 0.001; do
  NT=2 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/run_kprop.py $n 65 $s k2,k3f results/kprop
done; done
