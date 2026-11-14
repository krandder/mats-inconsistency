#!/usr/bin/env bash
# E71 (amendment 79): every 5 minutes queue evaluations of finished E71 nets (e71_queue.py); exit when all 36 nets have an
# evaluation output and a head file, then touch /tmp/s7q/e71_eval.done. usage: e71_loop.sh
cd /home/agent/workspaces/mats/work/cm-interp-B
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python e71_queue.py >> results/e71_queue.log 2>&1
  n=$(ls results/e71/e71_*.npz 2>/dev/null | wc -l)
  h=$(ls results/e71_heads/e71_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 36 ] && [ "$h" -ge 36 ]; then
    touch /tmp/s7q/e71_eval.done
    echo "$(date -u +%TZ) all 36 E71 nets evaluated" >> results/e71_queue.log
    break
  fi
  sleep 300
done
