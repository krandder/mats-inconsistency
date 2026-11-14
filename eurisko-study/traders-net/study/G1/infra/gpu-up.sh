#!/bin/bash
# Create one traders-net spot L4 VM for a queue job and wait until /var/lib/tn/READY.
# Usage: infra/gpu-up.sh --job ID --ws WS --hours H [--name NAME] [--no-wait]
# Refuses if: >= MAX_VMS traders-net VMs exist (exit 75), the job already has a VM, ledger + reserve
# > $BUDGET_USD or the workstream's spent + reserve > its cap in caps.tsv (exit 3). GCE deletes the VM
# at max-run-duration (H hours + setup margin, never more than 6 h). Last stdout line: "NAME ZONE IP".
# Exit 7 = the spot VM was preempted before READY.
set -eu
source "$(dirname "$0")/lib.sh"
JOB=""; WS=""; HOURS=""; NAME=""; WAIT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job) JOB=$2; shift ;; --ws) WS=$2; shift ;; --hours) HOURS=$2; shift ;;
    --name) NAME=$2; shift ;; --no-wait) WAIT=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac; shift
done
valid_id "$JOB" || { echo 'invalid --job' >&2; exit 2; }
[[ $WS =~ ^[A-Z][A-Z0-9]{1,3}$ ]] || { echo 'invalid --ws' >&2; exit 2; }
[[ $HOURS =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'invalid --hours' >&2; exit 2; }
MAX_MIN=$(max_run_min "$HOURS")
RATE=$(rate_for "$MACHINE")
RESERVE=$(awk -v r="$RATE" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", r*m/60}')

exec 8>"$STATE/admission.lock"; flock 8
existing=$(tn_vms) || { echo 'cannot list instances' >&2; exit 1; }
n=$(grep -c . <<<"$existing" || true)
(( n < MAX_VMS )) || { echo "WAIT: $n traders-net VMs exist (max $MAX_VMS)" >&2; exit 75; }
if grep -q -- "-$JOB-" <<<"$existing"; then echo "REFUSE: job $JOB already has a VM" >&2; exit 1; fi
SPENT=$(spend_usd)
if awk -v s="$SPENT" -v r="$RESERVE" -v b="$BUDGET_USD" 'BEGIN{exit !(s+r>b)}'; then
  echo "REFUSE: ledger \$$SPENT + reserve \$$RESERVE > cap \$$BUDGET_USD" >&2; exit 3; fi
WSP=$(spend_usd "$WS"); CAP=$(ws_cap "$WS")
if awk -v s="$WSP" -v r="$RESERVE" -v b="$CAP" 'BEGIN{exit !(s+r>b)}'; then
  echo "REFUSE: $WS ledger \$$WSP + reserve \$$RESERVE > $WS cap \$$CAP (infra/caps.tsv)" >&2; exit 3; fi

ensure_key
NAME=${NAME:-$(new_vm_name "$JOB")}
[[ $NAME =~ ^tn-gpu-[a-z0-9-]+$ && ${#NAME} -le 63 ]] || exit 2
cleanup_up() { local rc=$?; [[ $rc == 0 ]] || "$INFRA/gpu-down.sh" --only "$NAME" --reason "gpu-up failed rc=$rc" || true; exit "$rc"; }
trap cleanup_up EXIT
trap 'exit 130' INT TERM HUP
tmp=$(mktemp -d /tmp/tn-gpu-up.XXXXXX)
printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
log_gpu "PENDING-CREATE vm=$NAME type=$MACHINE rate_usd_h=$RATE reserve_usd=$RESERVE max_run_min=$MAX_MIN ws=$WS job=$JOB spent_before_usd=$SPENT"
ZONE=""
mapfile -t ZONES < <(printf '%s\n' "${ZONES[@]}" | shuf)   # spread spot load; smoke saw 2 preemptions in -a
for z in "${ZONES[@]}"; do
  printf '%s\n' "$z" >"$STATE/intent-$NAME"
  echo "creating $NAME ($MACHINE + $ACCEL, SPOT, max ${MAX_MIN}m) in $z ..." >&2
  if "${GC[@]}" compute instances create "$NAME" --zone="$z" \
      --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
      --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="${MAX_MIN}m" \
      --maintenance-policy=TERMINATE --no-restart-on-failure \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-balanced --boot-disk-auto-delete \
      --network=default --no-service-account --no-scopes \
      --labels="$LABEL=1,tn-job=$JOB,tn-ws=${WS,,}" \
      --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
      --metadata-from-file="startup-script=$INFRA/startup.sh,ssh-keys=$tmp/ssh-keys" >&2 2>"$tmp/err"; then
    ZONE=$z; break
  fi
  cat "$tmp/err" >&2
  grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available in zone' "$tmp/err" || break
  cat "$tmp/err" >>"$tmp/err.all"
done
if [[ -z $ZONE ]]; then
  log_gpu "CREATE-FAILED vm=$NAME ws=$WS job=$JOB zones=${ZONES[*]} err=\"$(cat "$tmp/err" 2>/dev/null | tr '\n' ' ' | cut -c1-200)\""
  "$INFRA/gpu-down.sh" --only "$NAME" --reason create-failed >/dev/null 2>&1 || true
  trap - EXIT; exit 4
fi
log_gpu "CREATE vm=$NAME zone=$ZONE type=$MACHINE gpu=1x$ACCEL model=SPOT max_run_min=$MAX_MIN rate_usd_h=$RATE ($RATE_NOTE) reserve_usd=$RESERVE ws=$WS job=$JOB"
flock -u 8; exec 8>&-
post_status "traders-net: GPU VM $NAME ($WS/$JOB) created in $ZONE, spot L4, ceiling \$$RATE/h, GCE deletes it after ${MAX_MIN}m"
IP=$(vm_ip "$NAME" "$ZONE")
if [[ $WAIT == 0 ]]; then trap - EXIT; echo "$NAME $ZONE $IP"; exit 0; fi
echo "waiting for $NAME ($IP) startup (up to 25 min) ..." >&2
deadline=$(( $(date +%s) + 1500 ))
while (( $(date +%s) < deadline )); do
  st=$(vm_ssh "$IP" 'cat /var/lib/tn/READY 2>/dev/null || { [ -f /var/lib/tn/FAILED ] && echo FAILED: $(cat /var/lib/tn/FAILED); }' 2>/dev/null || true)
  if [[ $st == FAILED* ]]; then log_gpu "STARTUP-FAILED vm=$NAME $st"; exit 5; fi
  if [[ -z $st ]] && ! "${GC[@]}" compute instances describe "$NAME" --zone="$ZONE" --format='value(status)' >/dev/null 2>&1 \
     && ! tn_vms | grep -q "^$NAME "; then
    log_gpu "PREEMPTED-AT-STARTUP vm=$NAME (instance gone before READY)"; exit 7
  fi
  if [[ -n $st ]]; then log_gpu "READY vm=$NAME ip=$IP $(tr '\n' ' ' <<<"$st")"; trap - EXIT; echo "$NAME $ZONE $IP"; exit 0; fi
  sleep 20
done
log_gpu "STARTUP-TIMEOUT vm=$NAME after 25 min"; exit 6
