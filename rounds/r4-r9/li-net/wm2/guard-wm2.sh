#!/bin/bash
# smallnet GPU guard (eurisko): delete any smallnet VM older than 250 min (4.2 h) and kill the youngest smallnet VM when audited + running estimate > $190.
# 2026-10-10 fix: the audited-cost grep matched only dates 2026-10-0x, so DELETE lines from 10-10 on were never counted.
LOG=/home/agent/workspaces/eurisko-study/lir-agent/GUARD-wm2.log
GPULOG=/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md
RATE=0.87
while true; do
  now=$(date -u +%s)
  audited=$(grep -E "^- 20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T.*(DELETE|COST-AUDIT) vm=soup-gpu-wm2" "$GPULOG" 2>/dev/null | grep -oE "cost_usd=[0-9.]+" | cut -d= -f2 | paste -sd+ | bc 2>/dev/null); audited=${audited:-0}
  running=0; names=""
  while IFS=$'\t' read -r name zone ts; do
    [ -z "$name" ] && continue
    c=$(date -u -d "$ts" +%s 2>/dev/null || echo "$now"); age=$(( (now - c) / 60 ))
    h=$(echo "scale=3; $age/60" | bc); running=$(echo "$running + $h*$RATE" | bc)
    names="$names $name:${age}m"
    if [ "$age" -gt 250 ]; then
      echo "$(date -u +%FT%TZ) AGE-KILL $name age=${age}m zone=$zone" >> "$LOG"
      gcloud compute instances delete "$name" --zone "$zone" --quiet >/dev/null 2>&1 && echo "$(date -u +%FT%TZ) deleted $name" >> "$LOG"
    fi
  done < <(gcloud compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-wm2' --format='value(name,zone.basename(),creationTimestamp)' 2>/dev/null)
  est=$(echo "$audited + $running" | bc)
  echo "$(date -u +%FT%TZ) STATUS audited=$audited running_est=$running total_est=$est vms:$names" >> "$LOG"
  if [ "$(echo "$est > 95" | bc)" = "1" ]; then
    youngest=$(gcloud compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-wm2' --sort-by=~creationTimestamp --format='value(name,zone.basename())' 2>/dev/null | head -1)
    if [ -n "$youngest" ]; then set -- $youngest; echo "$(date -u +%FT%TZ) CAP-KILL $1 est=$est" >> "$LOG"; gcloud compute instances delete "$1" --zone "$2" --quiet >/dev/null 2>&1; fi
  fi
  sleep 300
done
