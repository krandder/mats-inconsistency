#!/usr/bin/env bash
# Every 5 minutes, queue evaluation jobs for an episode's finished nets (queue_evals.py); when N nets are evaluated, queue the
# summary command SUMM (priority SPRIO) and exit. usage: evq_loop.sh EP EVAL.py PRIO N "SUMM" SPRIO
cd /home/agent/workspaces/mats/work/cm-interp-B
EP=$1; EVAL=$2; PRIO=$3; N=$4; SUMM=$5; SPRIO=$6
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python queue_evals.py "$EP" "$EVAL" "$PRIO" >> /tmp/s6q/evq_$EP.log 2>&1
  n=$(ls results/$EP/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge "$N" ]; then
    echo "$SUMM" > /tmp/s6q/pending/${SPRIO}_${EP}_summ
    echo "$(date -u +%TZ) all $N evaluated; summary queued" >> /tmp/s6q/evq_$EP.log
    break
  fi
  sleep 300
done
