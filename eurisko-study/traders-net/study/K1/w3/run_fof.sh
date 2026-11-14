#!/bin/bash
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HERE=/home/agent/workspaces/eurisko-study/traders-net/study/K1/w3
for N in 1000 10000; do for s in 1 2 3; do echo "$N $s"; done; done |
  xargs -P 3 -L 1 bash -c 'python3 '$HERE'/fof.py --rule fof --topo cross --seed $1 --N $0 --T 10000 > /tmp/traders-net/K1w3/logs/fof-N$0-s$1.log 2>&1'
