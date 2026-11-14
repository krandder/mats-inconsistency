#!/bin/bash
D=/home/agent/data/qwen06b-fixed-20260923
while true; do
  bash $D/ops/sync_once.sh
  grep -q QUEUE_DONE $D/results/status.txt 2>/dev/null && { echo SYNC_FINAL > $D/ops/sync-final.txt; break; }
  sleep 300
done
