#!/bin/bash
cd /home/agent/workspaces/eurisko-study/traders-net/study/M6/w3
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
for s in 0 1 2; do for a in a b kwd kboost elog eexp fb01 fb05; do echo "$a $s"; done; done | \
  xargs -P 3 -n 2 sh -c '[ -f results/$0-s$1.json ] || python3 m6w3.py $0 $1 >> results/run.log 2>&1'
echo ALLDONE >> results/run.log
