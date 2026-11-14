#!/usr/bin/env bash
# Every 5 minutes, queue evaluation jobs for finished E53 nets (e53_queue_evals.py); stop when all 40 are evaluated, then queue
# the summary (priority 30) and exit.
cd /home/agent/workspaces/mats/work/cm-interp-B
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python e53_queue_evals.py >> /tmp/s6q/evq.log 2>&1
  n=$(ls results/e53/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 40 ]; then
    echo "/home/agent/.venvs/aggext/bin/python e53_summ.py > results/e53_summ.out 2>&1 && touch /tmp/s6q/e53.done" > /tmp/s6q/pending/30_e53_summ
    echo "$(date -u +%TZ) all 40 evaluated; summary queued" >> /tmp/s6q/evq.log
    break
  fi
  sleep 300
done
