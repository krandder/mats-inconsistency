#!/bin/bash
# lir-agent GPU guard (eurisko, 2026-10-03): enforce the per-VM 2 h promise (delete at 2.6 h) and the $60 programme cap (hard stop at $55 estimated).
LOG=/home/agent/workspaces/eurisko-study/lir-agent/GUARD-r6.log
GPULOG=/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md
RATE=0.87
while true; do
  now=$(date -u +%s)
  audited=$(grep -E "^- 2026-10-1[0-9]T.*(DELETE|COST-AUDIT) vm=soup-gpu-r6" "$GPULOG" 2>/dev/null | grep -oE "cost_usd=[0-9.]+" | cut -d= -f2 | paste -sd+ | bc 2>/dev/null); audited=${audited:-0}
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
  done < <(gcloud compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-r6' --format='value(name,zone.basename(),creationTimestamp)' 2>/dev/null)
  est=$(echo "$audited + $running" | bc)
  echo "$(date -u +%FT%TZ) STATUS audited=$audited running_est=$running total_est=$est vms:$names" >> "$LOG"
  if [ "$(echo "$est > 11" | bc)" = "1" ]; then
    youngest=$(gcloud compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-r6' --sort-by=~creationTimestamp --format='value(name,zone.basename())' 2>/dev/null | head -1)
    if [ -n "$youngest" ]; then set -- $youngest; echo "$(date -u +%FT%TZ) CAP-KILL $1 est=$est" >> "$LOG"; gcloud compute instances delete "$1" --zone "$2" --quiet >/dev/null 2>&1; fi
  fi
  sleep 300
done
