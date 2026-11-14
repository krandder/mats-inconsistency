#!/bin/bash
# usage: side_queue.sh jobsfile ; runs each job only when fewer than 3 lane python jobs are running (nice 19, idle IO)
cd /home/agent/workspaces/mats/work/input-bits
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/agent/.venvs/arc-noisy/bin/python
while read -r job; do
  [ -z "$job" ] && continue
  while [ "$(ps -eo args | grep -E '^/home/agent/.venvs/arc-noisy/bin/python (scratch/|/home/agent/workspaces/mats/work/stationary/scratch/partB)' | wc -l)" -ge 3 ]; do sleep 30; done
  nice -n 19 ionice -c3 bash -c "$PY $job" || echo "FAILED: $job"
done < "$1"
echo "side queue $1 finished $(date -u +%H:%M:%SZ)"
