#!/bin/bash
cd /home/agent/workspaces/eurisko-study/traders-net/study/M6/lead
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
for s in 1 2 3; do for a in wd boost; do echo "$a $s"; done; done | \
  xargs -P 3 -n 2 sh -c '[ -f results/$0-s$1.json ] || python3 pool_kc.py --arm $0 --seed $1 --T 60000 >> results/run.log 2>&1'
echo ALLDONE >> results/run.log
