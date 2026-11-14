#!/bin/bash
cd /home/agent/workspaces/mats/work/lyapunov-dimension
for job in "$@"; do
  set -- $job
  seed=$1; cut=$2; lps=$3
  nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/n2_relu.py --seed $seed --n 256 --L 64 --cut $cut --Lps $lps --paths 2048 --svpaths 16 --sigma 0.001 > logs/n2b_256_64_c${cut}_s${seed}.log 2>&1
done
