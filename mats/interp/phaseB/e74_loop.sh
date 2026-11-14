#!/usr/bin/env bash
# E74 (amendment 82): every 5 minutes queue evaluations of finished S = 6 nets (e74_queue.py); exit when all 24 have an output.
cd /home/agent/workspaces/mats/work/cm-interp-B
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python e74_queue.py >> results/e74_queue.log 2>&1
  n=$(ls results/e74/S6/*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 24 ]; then
    echo "$(date -u +%TZ) all 24 E74 nets evaluated" >> results/e74_queue.log
    break
  fi
  sleep 300
done
