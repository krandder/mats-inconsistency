#!/bin/bash
# Delete traders-net VMs and record their cost. Idempotent. Never touches VMs outside the
# traders-net label / tn-gpu-* prefix (the soup and wealth-net VMs are other lanes').
# Usage: infra/gpu-down.sh --only NAME [--reason TEXT]   |   infra/gpu-down.sh --all [--reason TEXT]
# After each delete it runs cost-audit.py (GCE operation receipts -> COST-AUDIT line in GPU-LOG.md)
# and regenerates COST.md.
set -u
source "$(dirname "$0")/lib.sh"
REASON="gpu-down.sh"; ONLY=""; ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1 ;; --only) ONLY="$2"; shift ;; --reason) REASON="$2"; shift ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $ONLY || $ALL == 1 ]] || { echo "need --only NAME or --all" >&2; exit 2; }
[[ -z $ONLY || $ONLY == tn-gpu-* ]] || { echo "REFUSE: $ONLY is not a traders-net VM" >&2; exit 2; }

audit() { python3 "$INFRA/cost-audit.py" "$1" >>"$STATE/audit.log" 2>&1 || echo "$(now_utc) audit pending for $1 (rerun infra/cost-audit.py --pending)" >>"$STATE/audit.log"; }

if ! vms=$(tn_vms); then
  if [[ -n $ONLY && -s $STATE/intent-$ONLY ]]; then
    zone=$(tail -1 "$STATE/intent-$ONLY")
    if "${GC[@]}" compute instances delete "$ONLY" --zone="$zone" </dev/null; then
      log_gpu "DELETE vm=$ONLY zone=$zone reason=list-unavailable (reserve retained until audit)"; audit "$ONLY"; exit 0
    fi
  fi
  echo "ERROR: cannot list instances; cleanup unverified" >&2; exit 1
fi
rc=0
while read -r name zone created mtype labelled status; do
  [[ -n $name ]] || continue
  [[ -z $ONLY || $name == "$ONLY" ]] || continue
  age=$(age_hours "$created"); rate=$(rate_for "$mtype"); rate=${rate:-0}
  cost=$(awk -v r="$rate" -v a="$age" 'BEGIN{printf "%.4f", r*a}')
  if "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null; then
    log_gpu "DELETE vm=$name zone=$zone type=$mtype age_h=$age cost_usd=$cost (rate ceiling x age) reason=\"$REASON\""
    post_status "traders-net: GPU VM $name deleted after ${age}h (~\$$cost ceiling), reason: $REASON"
  elif "${GC[@]}" compute instances describe "$name" --zone="$zone" >/dev/null 2>&1; then
    log_gpu "DELETE-FAILED vm=$name zone=$zone reason=\"$REASON\""
    post_status "traders-net: FAILED to delete GPU VM $name in $zone, check now"; rc=1; continue
  else
    log_gpu "DELETE vm=$name zone=$zone type=$mtype age_h=$age cost_usd=$cost (already gone) reason=\"$REASON\""
  fi
  audit "$name"
done <<<"$vms"
# A VM that never appeared in the listing (create failed or already gone) still gets an audit.
if [[ -n $ONLY ]] && ! grep -q "^$ONLY " <<<"$vms"; then echo "no live VM named $ONLY"; audit "$ONLY"; fi
python3 "$INFRA/cost-report.py" >/dev/null 2>&1 || true
exit $rc
