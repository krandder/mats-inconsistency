#!/bin/bash
# P18 plan automation (PREREG 4): every 5 min run plan.py extra (best alpha + 2 seeds, once the 9 alpha-sweep evals
# exist) and plan.py a5 (spend rule), until a5 is decided, STOP exists, or 30 min before the deletion deadline.
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P18-detached-mazes/gpu/common.sh
G=$D/gpu
PY=/home/agent/.venvs/trader-cat/bin/python
DL=$(date -u -d "$DEADLINE_DEL" +%s)
while true; do
  [ -f $G/STOP ] && { echo "$(date -u +%FT%TZ) STOP"; exit 0; }
  [ $(( DL - $(date -u +%s) )) -lt 1800 ] && { echo "$(date -u +%FT%TZ) deadline window, no more planning"; exit 0; }
  if [ -f $G/main_plan.json ]; then
    echo "$(date -u +%FT%TZ) extra: $($PY $G/plan.py extra 2>&1 | tail -1)"
    echo "$(date -u +%FT%TZ) a5: $($PY $G/plan.py a5 2>&1 | tail -1)"
    $PY -c "import json,sys; sys.exit(0 if 'a5' in json.load(open('$G/main_plan.json')) else 1)" && { echo "a5 decided, auto done"; exit 0; }
  fi
  sleep 300
done
