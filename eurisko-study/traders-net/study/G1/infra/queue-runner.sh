#!/bin/bash
# Dispatch queued GPU jobs: at most MAX_VMS (2) at once, oldest first, budget-checked.
# Usage: infra/queue-runner.sh --once      (cron runs this every 2 min; safe to run by hand)
#        infra/queue-runner.sh --dry-run   (print what it would do)
# Per pass: (1) jobs in running/ whose job-run.sh died are marked failed and their VM deleted;
# (2) pending jobs whose worst-case reserve would push the ledger above $BUDGET_USD or their
# workstream above its caps.tsv cap move to refused/ with REFUSED.txt; (3) the rest start in order.
set -u
source "$(dirname "$0")/lib.sh"
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
exec 7>"$STATE/runner.lock"; flock -n 7 || exit 0
[[ -f $STATE/PAUSED ]] && { echo "$(now_utc) paused (infra/state/PAUSED exists)"; exit 0; }

# (1) orphans
for d in "$QUEUE"/running/*/; do
  [[ -d $d ]] || continue; j=$(basename "$d")
  pid=$(cat "$d/runner.pid" 2>/dev/null || true)
  if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then continue; fi
  # give a just-started job 2 minutes to write its pid
  (( $(date +%s) - $(stat -c %Y "$d") > 120 )) || continue
  echo "$(now_utc) orphan $j (runner pid ${pid:-none} dead)"
  [[ $DRY == 1 ]] && continue
  vm=$(cat "$d/vm" 2>/dev/null || true)
  [[ -n $vm ]] && "$INFRA/gpu-down.sh" --only "$vm" --reason "orphaned job $j"
  echo "$(now_utc) marked failed by queue-runner: runner process died" >>"$d/job.log"
  mv "$d" "$QUEUE/failed/$j"
done

running=$(find "$QUEUE/running" -mindepth 1 -maxdepth 1 -type d | wc -l)
live=$(tn_vms | grep -c . || true)
slots=$(( MAX_VMS - (running > live ? running : live) ))
spent=$(spend_usd)
# (2)+(3) pending, oldest job.env first
while read -r _ env; do
  [[ -n ${env:-} ]] || continue
  d=$(dirname "$env"); j=$(basename "$d")
  getv() { sed -n "s/^$1=//p" "$d/job.env" | tail -1 | tr -d '"'"'"; }
  ws=$(getv WS); h=$(getv HOURS)
  why=""
  valid_id "$j" || why="invalid job id"
  [[ -f $d/run.sh ]] || why="no run.sh"
  [[ $h =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v h="$h" 'BEGIN{exit !(h>0 && h<=5.5)}' || why="${why:+$why; }HOURS must be in (0, 5.5]"
  if [[ -z $why ]]; then
    r=$(reserve_usd "$h"); cap=$(ws_cap "$ws"); wsp=$(spend_usd "$ws")
    awk -v s="$spent" -v r="$r" -v b="$BUDGET_USD" 'BEGIN{exit !(s+r>b)}' && why="ledger \$$spent + reserve \$$r > cap \$$BUDGET_USD"
    awk -v s="$wsp" -v r="$r" -v b="$cap" 'BEGIN{exit !(s+r>b)}' && why="${why:+$why; }$ws ledger \$$wsp + reserve \$$r > $ws cap \$$cap"
  fi
  if [[ -n $why ]]; then
    echo "$(now_utc) refuse $j: $why"
    [[ $DRY == 1 ]] && continue
    echo "$(now_utc) REFUSED by queue-runner: $why" >>"$d/REFUSED.txt"; mv "$d" "$QUEUE/refused/$j"; continue
  fi
  (( slots > 0 )) || { echo "$(now_utc) waiting: $j (no free VM slot)"; continue; }
  echo "$(now_utc) start $j ws=$ws hours=$h reserve=\$$r"
  [[ $DRY == 1 ]] && { slots=$((slots-1)); continue; }
  mv "$d" "$QUEUE/running/$j" && touch "$QUEUE/running/$j"
  setsid nohup "$INFRA/job-run.sh" "$j" </dev/null >>"$STATE/runner.log" 2>&1 &
  slots=$((slots-1)); spent=$(awk -v s="$spent" -v r="$r" 'BEGIN{printf "%.2f", s+r}')
  sleep 5
done < <(find "$QUEUE/pending" -mindepth 2 -maxdepth 2 -name job.env -printf '%T@ %p\n' 2>/dev/null | sort -n)
exit 0
