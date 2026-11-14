#!/bin/bash
# NC1/NC2 evaluation queue (PREREG.md), at most 3 concurrent, nice 19 / ionice idle.
cd /home/agent/workspaces/mats/work/cut-rank-ext
PY=/home/agent/.venvs/arc-noisy/bin/python
echo "queue start $(date -u +%H:%M:%SZ)" >> logs/nc1_queue.log
cat scratch/q_nc1.txt | xargs -P 3 -L 1 bash -c 'set -- "$@"; nm="nl_n$1_c$2_d$3"; nice -n 19 ionice -c3 '"$PY"' scratch/nl_readout.py "$@" > logs/$nm.log 2>&1; echo "$nm exit $? $(date -u +%H:%M:%SZ)" >> logs/nc1_queue.log' _
echo "queue done $(date -u +%H:%M:%SZ)" >> logs/nc1_queue.log
