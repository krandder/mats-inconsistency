#!/bin/bash
# Stop-guarantee layer 2: runs from cron every 15 min on General.
# Lists logical-soup VMs (label logical-soup, or name soup-gpu-*) in GCP_PROJECT,
# posts one line per VM per hour to Fleet - Status, and deletes (via gpu-down.sh)
# any VM older than 8 h or named soup-gpu-* without the label. Events go to GPU-LOG.md.
# Usage: infra/gpu-watchdog.sh [--dry-run]
set -u
source "$(dirname "$0")/lib.sh"
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
MAX_AGE_H=8

exec 9>"$STATE/watchdog.lock"
flock -n 9 || { echo "$(now_utc) previous watchdog still running"; exit 0; }

if ! vms=$(soup_vms); then
  # Listing failed: the guarantee is degraded. Alert at most every 6 h.
  f="$STATE/watchdog-list-failed"; last=$(cat "$f" 2>/dev/null || echo 0)
  if (( $(date +%s) - last > 21600 )); then
    [[ $DRY == 1 ]] || post_status "logical-soup gpu-watchdog: cannot list GCE instances in $PROJECT (gcloud failed); stop guarantee degraded"
    date +%s >"$f"
  fi
  echo "$(now_utc) list FAILED"; exit 1
fi
rm -f "$STATE/watchdog-list-failed"

if [[ -z "$vms" ]]; then echo "$(now_utc) ok: no logical-soup VM"; exit 0; fi

while read -r name zone created mtype labelled status; do
  [[ -n $name ]] || continue
  age=$(age_hours "$created")
  why=""
  awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="older than ${MAX_AGE_H}h (${age}h)"
  [[ $labelled == no && $name == soup-gpu-* ]] && why="${why:+$why; }named soup-gpu-* but no $LABEL label"
  echo "$(now_utc) seen $name zone=$zone type=$mtype age_h=$age labelled=$labelled status=$status ${why:+DELETE: $why}"
  if [[ -n $why ]]; then
    if [[ $DRY == 1 ]]; then echo "dry-run: would delete $name ($why)"; continue; fi
    log_gpu "WATCHDOG vm=$name age_h=$age action=delete why=\"$why\""
    "$INFRA/gpu-down.sh" --only "$name" --reason "watchdog: $why"
    continue
  fi
  f="$STATE/watchdog-posted-$name"; last=$(cat "$f" 2>/dev/null || echo 0)
  if (( $(date +%s) - last >= 3600 )); then
    rate=$(rate_for "$mtype")
    msg="logical-soup: GPU VM $name up ${age}h in $zone ($mtype, $status, ceiling \$${rate:-?}/h); self-deletes at 6h, watchdog deletes at ${MAX_AGE_H}h"
    if [[ $DRY == 1 ]]; then echo "dry-run: would post: $msg"; continue; fi
    post_status "$msg"
    log_gpu "WATCHDOG vm=$name age_h=$age status=$status action=posted"
    date +%s >"$f"
  fi
done <<<"$vms"
