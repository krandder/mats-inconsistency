#!/bin/bash
# Run a queue of run_cut.py configurations, at most P at a time, at the lowest CPU and IO priority.
Q=$1; P=${2:-4}
cd /home/agent/workspaces/mats/work/cut-rank
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -v '^\s*$' "$Q" | xargs -P "$P" -I{} sh -c 'nice -n 19 ionice -c3 python3 scratch/run_cut.py {} >> logs/queue.out 2>> logs/queue.err; echo "done {} $(date -u +%H:%M:%SZ)" >> logs/queue.done'
echo "queue $Q finished $(date -u +%H:%M:%SZ)" >> logs/queue.done
