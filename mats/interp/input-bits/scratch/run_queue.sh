#!/bin/bash
# usage: run_queue.sh jobsfile  (one shell command per line; at most 3 at a time; nice 19, idle IO, 1 BLAS thread)
cd /home/agent/workspaces/mats/work/input-bits
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/agent/.venvs/arc-noisy/bin/python
grep -v '^#' "$1" | grep -v '^$' | xargs -P 3 -I{} bash -c "nice -n 19 ionice -c3 $PY {} || echo FAILED: {}"
echo "queue $1 finished $(date -u +%H:%M:%SZ)"
