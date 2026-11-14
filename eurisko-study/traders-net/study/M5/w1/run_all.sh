#!/bin/bash
# M5 w1 job queue: 2 single-thread jobs at once (PREREG operational rule).
cd "$(dirname "$0")"
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=""
for s in 1 2 3; do for st in 8000 16000 30000; do jobs+="wn-ckpt$st $s\n"; done; done
for src in P4-net P16-net; do for s in 1 2 3; do jobs+="$src $s\n"; done; done
jobs+="P4-logistic 2\nP4-logistic 3\n"
printf "$jobs" | while read src s; do [ -f results/$src-s$s.json ] || echo "$src $s"; done | \
  xargs -P 2 -L 1 bash -c 'timeout 1800 python3 calib.py $0 $1 >> /tmp/traders-net/M5/w1/run.log 2>&1 || echo "FAIL $0 $1" >> /tmp/traders-net/M5/w1/run.log'
echo ALLDONE >> /tmp/traders-net/M5/w1/run.log
