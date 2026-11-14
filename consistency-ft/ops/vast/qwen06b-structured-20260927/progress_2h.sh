#!/bin/bash
# Append a status line to the calibrated-ai progress log every 2 h until the instance is destroyed.
D=/home/agent/data/qwen06b-structured-20260927; L=/home/agent/workspaces/calibrated-ai/qwen06b-progress.log
while [ ! -e $D/ops/destroyed.json ]; do
  sleep 7200; [ -e $D/ops/destroyed.json ] && break
  done_arms=$(grep -c 'EXIT=' $D/results/status.txt 2>/dev/null)
  running=$(grep START $D/results/status.txt 2>/dev/null | awk '{print $1}' | while read a; do grep -q "^$a EXIT" $D/results/status.txt || echo $a; done | tr '\n' ' ')
  spent=$(python3 -c "import json;print(round(json.load(open('$D/ops/spend.json'))['spent_est'],2))" 2>/dev/null)
  echo "$(date -u +%FT%TZ) qwen06b-structured 2h check: $done_arms/5 arms done; running: $running; est spend \$$spent; last sync $(cat $D/ops/last-sync.txt 2>/dev/null)" >> $L
done
