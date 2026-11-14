#!/bin/bash
# run queued X1w1 jobs one at a time when fewer than 3 of my jobs run
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1
mine() { pgrep -f "python3 (run_w1|check_|audit|baseline|null_calib|summarize|aggregate)" -a | while read pid rest; do d=$(readlink /proc/$pid/cwd 2>/dev/null); case "$d" in /tmp/traders-net/X1w1/*) echo $pid;; esac; done | wc -l; }
while read -r job; do
  [ -z "$job" ] && continue
  while [ "$(mine)" -ge 3 ]; do sleep 15; done
  echo "$(date -u +%FT%TZ) start: $job"
  bash -c "$job" &
  sleep 5
done < /tmp/traders-net/X1w1/queue2.txt
wait; date -u > /tmp/traders-net/X1w1/queue2_done.flag
