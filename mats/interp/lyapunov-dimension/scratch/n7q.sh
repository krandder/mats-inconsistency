#!/bin/bash
cd /home/agent/workspaces/mats/work/lyapunov-dimension
for job in "$@"; do
  set -- $job
  OMP_NUM_THREADS=2 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/n7_pctail.py --seed $1 --n $2 --L $3 --cut $4 --paths 8192 --probes 2 --statepaths 65536 >> logs/n7_queue.log 2>&1
done
