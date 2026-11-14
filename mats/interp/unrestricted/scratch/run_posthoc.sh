#!/bin/bash
# Post-hoc jobs (not registered), run after the evaluation queue: 3 at a time, nice 19, idle IO.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/agent/workspaces/mats/work/unrestricted
until grep -qE "^queue end [0-9]" logs/queue.log; do sleep 30; done
echo "posthoc start $(date -u +%H:%M:%SZ)" >> logs/queue.log
xargs -P 3 -I{} -d '\n' nice -n 19 ionice -c3 bash -c '{}' < scratch/jobs_posthoc.txt
echo "posthoc end $(date -u +%H:%M:%SZ)" >> logs/queue.log
