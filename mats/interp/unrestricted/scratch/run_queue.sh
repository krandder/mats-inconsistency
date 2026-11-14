#!/bin/bash
# Evaluation queue: 3 jobs at a time, nice 19, idle IO, single-threaded BLAS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/agent/workspaces/mats/work/unrestricted
echo "queue start $(date -u +%H:%M:%SZ)" >> logs/queue.log
xargs -P 3 -I{} -d '\n' nice -n 19 ionice -c3 bash -c '{}' < scratch/jobs_eval2.txt
echo "queue end $(date -u +%H:%M:%SZ)" >> logs/queue.log
