#!/bin/bash
cd /home/agent/workspaces/mats/work/sparse-codes
for s in 0.01 0.001; do
  NT=2 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/run_kprop.py 256 65 $s k2,k3f results/kprop
done
