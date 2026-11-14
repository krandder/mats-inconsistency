#!/bin/bash
# K1 worker3 grid: rules x topologies x seeds at N (arg 1, default 1000); 3 concurrent single-thread jobs.
N=${1:-1000}; T=${2:-10000}
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HERE=/home/agent/workspaces/eurisko-study/traders-net/study/K1/w3
mkdir -p /tmp/traders-net/K1w3/logs
for s in 1 2 3; do for r in bh crp eg; do for tp in within cross; do echo "$r $tp $s"; done; done; done |
  xargs -P 3 -L 1 bash -c 'python3 '$HERE'/fof.py --rule $0 --topo $1 --seed $2 --N '$N' --T '$T' > /tmp/traders-net/K1w3/logs/$0-$1-N'$N'-s$2.log 2>&1'
