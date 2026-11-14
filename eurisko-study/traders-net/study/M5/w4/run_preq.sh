#!/bin/bash
# preq-P4/P16 arms (PREREG A2); shares the 3-job cap with run_all.sh via pgrep; load-gated <40.
D=/home/agent/workspaces/eurisko-study/traders-net/study/M5/w4
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
L=/tmp/traders-net/M5w4/logs; mkdir -p $L
for c in P4 P16; do for s in 1 2 3; do
  while [ $(pgrep -f "M5/w4/probes.py" | wc -l) -ge 3 ]; do sleep 10; done
  while [ $(cut -d. -f1 /proc/loadavg) -ge 40 ]; do sleep 20; done
  nice -n 10 python3 $D/probes.py --source preq --ckpt $c --seed $s > $L/preq_${c}_s$s.log 2>&1 &
  sleep 5
done; done
wait; echo PREQDONE
