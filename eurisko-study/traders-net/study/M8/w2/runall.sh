#!/bin/bash
# Runs every arm x seed, 4 at a time, 1 thread each. Logs to /tmp/traders-net/M8w2/.
cd /tmp/traders-net/M8w2
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1
G=/home/agent/workspaces/eurisko-study/traders-net/study/M8/w2/gated.py
for s in 1 2 3; do for a in absag random taylor movement magnitude cbp static128 dense256; do echo "$a $s"; done; done |
  xargs -P 4 -n 2 sh -c 'python3 '"$G"' "$0" "$1" > log_$0_s$1.txt 2>&1'
