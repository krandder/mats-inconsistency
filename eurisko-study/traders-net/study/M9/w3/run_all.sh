#!/bin/bash
# 42 runs, 3 concurrent; skips finished runs. Order: seed-major so partial results are balanced.
cd "$(dirname "$0")"
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
for s in 1 2 3; do for arm in B A; do for kh in "16 8" "1 8" "4 8" "64 8" "1 128" "4 32" "64 2"; do
  set -- $kh; [ -f results/$arm-K$1-hw$2-s$s.json ] || echo "$arm $1 $2 $s"; done; done; done |
xargs -P 3 -L 1 sh -c 'while [ "$(cut -d" " -f1 /proc/loadavg | cut -d. -f1)" -gt 40 ]; do sleep 30; done; python3 run_kq.py $0 $1 $2 $3' >> results/run.log 2>&1
