#!/bin/bash
# E0 descriptive queue (CPU): args = seeds (256 x 64, 2^17 paths)
cd /home/agent/workspaces/mats/work/nongauss-family
for s in "$@"; do
  TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/e0.py $s 17 256 64 0.001 --out runs/e0 > logs/e0_256_64_${s}.log 2>&1
done
