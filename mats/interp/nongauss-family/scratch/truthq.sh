#!/bin/bash
# sequential truth queue: args = seeds (width 256, depth 64, sigma 0.001, 2^22 paths)
cd /home/agent/workspaces/mats/work/nongauss-family
for s in "$@"; do
  TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/truth.py $s 22 256 64 0.001 > logs/truth_256_64_${s}.log 2>&1
done
