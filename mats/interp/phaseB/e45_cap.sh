#!/usr/bin/env bash
# Amendment 53 spend cap: stop every E45 API stream if the Claude list-price total in results/e45_calls.jsonl passes $6.
cd /home/agent/workspaces/mats/work/cm-interp-B
while true; do
  cost=$(/home/agent/.venvs/llmexp/bin/python -c "
import json, os
c = 0.0
if os.path.exists('results/e45_calls.jsonl'):
    for l in open('results/e45_calls.jsonl'): c += json.loads(l).get('cost') or 0
print(c)")
  if python3 -c "import sys; sys.exit(0 if $cost > 6 else 1)"; then
    echo "$(date -u) cap reached: cost $cost; stopping E45 API streams" >> results/e45_cap.log
    pkill -f 'e45_call.py [SU]_(haiku45|sonnet46|opus48|gpt56)' ; exit 0
  fi
  if ! pgrep -f 'e45_call.py' > /dev/null; then echo "$(date -u) no E45 streams left; cost $cost" >> results/e45_cap.log; exit 0; fi
  sleep 60
done
