#!/usr/bin/env bash
# Amendment 25 spend cap: stop every E18 stream if the Claude list-price total passes $25 or the calls pass 1,200.
cd /home/agent/workspaces/mats/work/cm-interp-B
while pgrep -f "e18_call.py" >/dev/null; do
  read cost calls < <(/home/agent/.venvs/llmexp/bin/python -c "
import json
c=n=0
for l in open('results/e18_calls.jsonl'):
    r=json.loads(l); c+=r.get('cost') or 0; n+=1
print('%.4f %d' % (c, n))")
  if python3 -c "import sys; sys.exit(0 if ($cost > 25 or $calls > 1200) else 1)"; then
    echo "$(date -u) cap reached: cost $cost calls $calls; stopping E18 streams"
    pkill -f "for a in S_opus R_opus"; pkill -f "for a in S_sonnet U_sonnet"; pkill -f "kill -0 250961"; pkill -f "e18_call.py"
    break
  fi
  sleep 60
done
