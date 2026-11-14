#!/bin/bash
# Stop-guarantee layer 2 (cron every 15 min). Lists traders-net VMs only (label traders-net or
# name tn-gpu-*), posts one line per VM per hour to Fleet - Status, and deletes any VM that is
# older than 6.25 h, lacks the label, or (after 30 min) has no job in queue/running/ naming it.
# Then audits pending costs and regenerates COST.md.
# Usage: infra/gpu-watchdog.sh [--dry-run]
set -u
source "$(dirname "$0")/lib.sh"
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
MAX_AGE_H=6.25
exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "$(now_utc) previous watchdog still running"; exit 0; }
if ! vms=$(tn_vms); then
  f="$STATE/watchdog-list-failed"; last=$(cat "$f" 2>/dev/null || echo 0)
  if (( $(date +%s) - last > 21600 )); then
    [[ $DRY == 1 ]] || post_status "traders-net gpu-watchdog: cannot list GCE instances (gcloud failed); stop guarantee degraded"
    date +%s >"$f"
  fi
  echo "$(now_utc) list FAILED"; exit 1
fi
rm -f "$STATE/watchdog-list-failed"
owned=$(cat "$QUEUE"/running/*/vm 2>/dev/null || true)
if [[ -z $vms ]]; then echo "$(now_utc) ok: no traders-net VM"; fi
while read -r name zone created mtype labelled status; do
  [[ -n $name ]] || continue
  age=$(age_hours "$created"); why=""
  awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="older than ${MAX_AGE_H}h (${age}h)"
  [[ $labelled == no ]] && why="${why:+$why; }no $LABEL label"
  if ! grep -qx "$name" <<<"$owned" && awk -v a="$age" 'BEGIN{exit !(a>0.5)}'; then why="${why:+$why; }no running job owns it"; fi
  echo "$(now_utc) seen $name zone=$zone age_h=$age status=$status ${why:+DELETE: $why}"
  if [[ -n $why ]]; then
    [[ $DRY == 1 ]] && { echo "dry-run: would delete $name ($why)"; continue; }
    log_gpu "WATCHDOG vm=$name age_h=$age action=delete why=\"$why\""
    "$INFRA/gpu-down.sh" --only "$name" --reason "watchdog: $why"; continue
  fi
  f="$STATE/watchdog-posted-$name"; last=$(cat "$f" 2>/dev/null || echo 0)
  if (( $(date +%s) - last >= 3600 )) && [[ $DRY == 0 ]]; then
    post_status "traders-net: GPU VM $name up ${age}h in $zone ($status, ceiling \$0.87/h); ledger \$$(spend_usd) of \$$BUDGET_USD"
    date +%s >"$f"
  fi
done <<<"$vms"
[[ $DRY == 1 ]] || { python3 "$INFRA/cost-audit.py" --pending >>"$STATE/audit.log" 2>&1; python3 "$INFRA/cost-report.py" >/dev/null 2>&1; } || true
