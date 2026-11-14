#!/bin/bash
cd /home/agent/workspaces/mats/work/sparse-codes
for n in 64 128 256; do for law in shift aniso std; do
  NT=2 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/run_kprop_law.py $n 65 0.01 $law k2,k3f results/kprop_law
done; done
