#!/usr/bin/env bash
# Amendment 32 spend cap: stop every E18b API stream if the Claude list-price total passes $10.
cd /home/agent/workspaces/mats/work/cm-interp-B
while ps -eo args | grep -q "[e]18b_call.py"; do
  cost=$(/home/agent/.venvs/llmexp/bin/python -c "
import json, os
c = 0
if os.path.exists('results/e18b_calls.jsonl'):
    for l in open('results/e18b_calls.jsonl'): c += json.loads(l).get('cost') or 0
print('%.4f' % c)")
  if python3 -c "import sys; sys.exit(0 if $cost > 10 else 1)"; then
    echo "$(date -u) cap reached: cost $cost; stopping E18b API streams"
    for p in $(ps -eo pid,args | awk '/[e]18b_call.py [SU]_(haiku45|sonnet46|opus48|gpt56)/ {print $1}'); do kill $p; done
    break
  fi
  sleep 60
done
