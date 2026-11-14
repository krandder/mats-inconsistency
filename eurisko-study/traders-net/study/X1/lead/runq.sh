#!/bin/bash
W=/tmp/traders-net/X1lead; MAXJ=${MAXJ:-3}
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export KQ_OUT=$W/w4/study/M9/w3/results KQ_TMP=$W/w4/M9w3
while IFS='|' read -r name cmd; do
  [ -z "$name" ] && continue
  [ -f $W/logs/$name.done ] && continue
  while [ $(jobs -r | wc -l) -ge $MAXJ ]; do sleep 10; done
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -gt 40 ]; do sleep 30; done
  ( s=$(date +%s); echo "$(date -u +%FT%TZ) START $name" >> $W/queue.log
    nice -n 10 bash -c "$cmd" > $W/logs/$name.log 2>&1; rc=$?
    echo "$(date -u +%FT%TZ) END $name rc=$rc wall=$(( $(date +%s)-s ))s" >> $W/queue.log
    [ $rc -eq 0 ] && touch $W/logs/$name.done ) &
  sleep 3
done < $W/jobs.txt
wait; echo "$(date -u +%FT%TZ) ALLDONE" >> $W/queue.log
