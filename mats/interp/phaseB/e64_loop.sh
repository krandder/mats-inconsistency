#!/usr/bin/env bash
# E64 (amendment 72): every 5 minutes queue evaluations of finished E64 nets (e64_queue.py); exit when the three code-share outputs
# exist, then touch /tmp/s7q/e64_codes.done. usage: e64_loop.sh
cd /home/agent/workspaces/mats/work/cm-interp-B
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python e64_queue.py >> results/e64_queue.log 2>&1
  n=$(ls results/e64_codes/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 3 ]; then
    touch /tmp/s7q/e64_codes.done
    echo "$(date -u +%TZ) all three code-share outputs exist" >> results/e64_queue.log
    break
  fi
  sleep 300
done
