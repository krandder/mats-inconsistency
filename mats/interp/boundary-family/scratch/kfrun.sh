#!/bin/bash
# K-F (PREREG amendment 1): nets 101, 102 at layers 16, 24, 31
cd /home/agent/workspaces/mats/work/boundary-family
for s in 101 102; do
  TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag8.py $s 16,24,31 17 > logs/kf_${s}.log 2>&1
done
