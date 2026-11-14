#!/bin/bash
# usage: slot_queue.sh jobsfile ; launches each job when fewer than 3 lane python jobs run (nice 19, idle IO, 1 BLAS thread)
cd /home/agent/workspaces/mats/work/input-bits
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/agent/.venvs/arc-noisy/bin/python
count() { ps -eo args | grep -E '^/home/agent/.venvs/arc-noisy/bin/python (scratch/|/home/agent/workspaces/mats/work/stationary/scratch/partB)' | wc -l; }
while read -r job; do
  [ -z "$job" ] && continue
  case "$job" in \#*) continue;; esac
  while [ "$(count)" -ge 3 ]; do sleep 15; done
  nohup nice -n 19 ionice -c3 bash -c "$PY $job || echo FAILED: $job" >> logs/slot_queue_failures.log 2>&1 &
  sleep 3
done < "$1"
while [ "$(count)" -gt 0 ]; do sleep 30; done
echo "slot queue $1 finished $(date -u +%H:%M:%SZ)"
