#!/bin/bash
# Create a logical-soup GPU VM (spot L4 by default) and wait until it is ready.
# Usage: infra/gpu-up.sh [--a100] [--zone ZONE] [--run-id ID] [--no-wait]
# Named rentals isolate each run; admission is serialized under the budget lock.
# Every VM self-deletes after 6 h (--max-run-duration=6h --instance-termination-action=DELETE).
set -eu
source "$(dirname "$0")/lib.sh"

MACHINE="g2-standard-8"; ACCEL="nvidia-l4"; ZONES=(us-central1-a us-central1-b us-central1-c)
[[ -z ${GPU_ZONES:-} ]] || read -r -a ZONES <<<"$GPU_ZONES"
# 2026-10-10 (smallnet/eurisko): GPU_PROVISIONING=STANDARD opts into on-demand VMs (no spot preemption; same
# $0.87/h ceiling rate used for budgeting). Default stays SPOT. Backup: /tmp/smallnet/gpu-up.sh.bak-1010.
PROV=${GPU_PROVISIONING:-SPOT}
[[ $PROV == SPOT || $PROV == STANDARD ]] || { echo "GPU_PROVISIONING must be SPOT or STANDARD" >&2; exit 2; }
WAIT=1; RUN_ID=""; NAME=""; GROUP=${GPU_BUDGET_GROUP:-}; GROUP_CAP=${GPU_GROUP_BUDGET_USD:-40}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --a100) MACHINE="a2-highgpu-1g"; ACCEL="nvidia-tesla-a100"; ZONES=(us-central1-a us-central1-b us-central1-c us-central1-f) ;;
    --name) NAME=$2; shift ;;
    --run-id) RUN_ID=$2; shift ;;
    --zone) ZONES=("$2"); shift ;;
    --no-wait) WAIT=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done

[[ -z $RUN_ID ]] || valid_run_id "$RUN_ID" || { echo 'invalid --run-id' >&2; exit 2; }
[[ -z $GROUP ]] || valid_run_id "$GROUP" || { echo 'invalid budget group' >&2; exit 2; }
[[ $GROUP_CAP =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
# Serialize admission and reservation, including concurrent named launches.
exec 8>"$STATE/admission.lock"
flock 8
existing=$(soup_vms) || { echo 'cannot list instances' >&2; exit 1; }
if [[ -z $RUN_ID && -n $existing ]]; then
  echo 'Use --run-id when other soup VMs exist' >&2; exit 1
fi
if [[ -n $RUN_ID ]]; then
  own=$(run_vms "$RUN_ID") || exit 1
  [[ -z $own ]] || { echo "run already has a VM: $own" >&2; exit 1; }
fi

RATE=$(rate_for "$MACHINE")
# Reservation = worst case for THIS VM: the run timeout (GPU_RUN_TIMEOUT, as gpu-run.sh enforces it) plus
# 0.6 h of margin (the lir-agent guard deletes any VM older than 2.6 h), capped at the 6 h max-run.
# Before 2026-10-03 11:20Z every VM reserved the full 6 h ($5.22), which refused launches at 9 live VMs.
RESERVE_H=$MAX_RUN_H
if [[ -n ${GPU_RUN_TIMEOUT:-} ]]; then
  th=$(awk -v t="$GPU_RUN_TIMEOUT" 'BEGIN{u=substr(t,length(t)); v=substr(t,1,length(t)-1)+0; if(u=="m")h=v/60; else if(u=="h")h=v; else if(u=="s")h=v/3600; else h=(t+0)/3600; if(h<=0)h=6; printf "%.2f", h+0.6}')
  awk -v a="$th" -v b="$MAX_RUN_H" 'BEGIN{exit !(a<b)}' && RESERVE_H=$th
fi
RESERVE=$(awk -v r="$RATE" -v h="$RESERVE_H" 'BEGIN{printf "%.2f", r*h}')
SPENT=$(spend_usd)
if awk -v s="$SPENT" -v r="$RESERVE" -v b="$BUDGET_USD" 'BEGIN{exit !(s+r>b)}'; then
  echo "REFUSE: spent \$$SPENT + this VM's worst case \$$RESERVE > budget \$$BUDGET_USD (GPU-LOG.md)" >&2; exit 1
fi

if [[ -n $GROUP ]]; then
  GROUP_SPENT=$(spend_usd "$GROUP")
  # The programme owner may raise a group's cap while runs are live: a number in $STATE/group-cap.<group>
  # (used only if larger than the cap the launcher passed). 2026-10-03 eurisko: lir-agent 50 -> 55 at 13:30Z, then 58 at 13:38Z, so that
  # E7's last wave is not starved by siblings' reservations; the night cap is $60 and guard.sh stops at $55.
  if [[ -f "$STATE/group-cap.$GROUP" ]]; then
    ov=$(tr -dc '0-9.' < "$STATE/group-cap.$GROUP")
    if [[ $ov =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v a="$ov" -v b="$GROUP_CAP" 'BEGIN{exit !(a>b)}'; then GROUP_CAP=$ov; fi
  fi
  if awk -v s="$GROUP_SPENT" -v r="$RESERVE" -v b="$GROUP_CAP" 'BEGIN{exit !(s+r>b)}'; then
    echo "REFUSE: group $GROUP reserved/spent $GROUP_SPENT + $RESERVE > $GROUP_CAP" >&2; exit 3
  fi
fi

"${GC[@]}" compute images describe-from-family "$IMAGE_FAMILY" --project="$IMAGE_PROJECT" --format='value(name)' >/dev/null \
  || { echo "REFUSE: image family $IMAGE_FAMILY not found in $IMAGE_PROJECT" >&2; exit 1; }

ensure_key
NAME=${NAME:-$(new_vm_name "$RUN_ID")}
[[ $NAME =~ ^soup-gpu-[a-z0-9-]+$ && ${#NAME} -le 63 ]] || exit 2
cleanup_up() {
  local rc=$?
  if [[ $rc != 0 ]]; then "$INFRA/gpu-down.sh" --only "$NAME" --no-sync --reason "gpu-up failed rc=$rc" || true; fi
  exit "$rc"
}
trap cleanup_up EXIT
trap 'exit 130' INT TERM HUP

tmp=$(mktemp -d /tmp/soup-gpu-up.XXXXXX)
# 2026-10-11 (wm2/eurisko): GPU_STARTUP=minimal opts into infra/startup-minimal.sh (driver check + rsync/curl only; no
# clang, bun or Bend2), for PyTorch-only jobs. Default unchanged (full). Backup: /tmp/wm2/gpu-up.sh.bak-1011.
STARTUP_SRC="$INFRA/startup.sh"
[[ ${GPU_STARTUP:-full} == minimal ]] && STARTUP_SRC="$INFRA/startup-minimal.sh"
sed -e "s|__BEND_REF__|$BEND_REF|g" -e "s|__BEND_LOCAL__|$BEND_LOCAL|g" -e "s|__BUN_VERSION__|$BUN_VERSION|g" \
  "$STARTUP_SRC" >"$tmp/startup.sh"
printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"

log_gpu "PENDING-CREATE vm=$NAME type=$MACHINE rate_usd_h=$RATE reserve_usd=$RESERVE group=${GROUP:-none} run=${RUN_ID:-default} by=gpu-up.sh"
ZONE=""
for z in "${ZONES[@]}"; do
  printf '%s\n' "$z" > "$STATE/intent-$NAME"
  echo "creating $NAME ($MACHINE + $ACCEL, $PROV) in $z ..."
  if "${GC[@]}" compute instances create "$NAME" --zone="$z" \
      --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
      --provisioning-model=$PROV --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
      --maintenance-policy=TERMINATE --no-restart-on-failure \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-balanced \
      --network=default --no-service-account --no-scopes \
      --labels="$LABEL=1${RUN_ID:+,soup-run=$RUN_ID}${GROUP:+,soup-group=$GROUP}" \
      --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
      --metadata-from-file="startup-script=$tmp/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
    ZONE=$z; break
  fi
  cat "$tmp/err" >&2
  grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available in zone|QUOTA' "$tmp/err" || break
done
if [[ -z "$ZONE" ]]; then
  log_gpu "CREATE-FAILED vm=$NAME type=$MACHINE zones=${ZONES[*]} err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
  # A failed create can still leave a half-made instance; clean it up.
  "$INFRA/gpu-down.sh" --only "$NAME" --no-sync >/dev/null 2>&1 || true
  exit 1
fi
log_gpu "CREATE vm=$NAME zone=$ZONE type=$MACHINE gpu=1x$ACCEL model=$PROV max_run=$MAX_RUN rate_usd_h=$RATE ($RATE_NOTE) reserve_usd=$RESERVE spent_before_usd=$SPENT group=${GROUP:-none} run=${RUN_ID:-default} by=gpu-up.sh"
echo "$NAME $ZONE" >"$STATE/vm-${RUN_ID:-default}"
flock -u 8
exec 8>&-
post_status "logical-soup: GPU VM $NAME created in $ZONE ($MACHINE, ${PROV,,}, rate ceiling \$$RATE/h, self-deletes after $MAX_RUN)"

[[ $WAIT == 1 ]] || { echo "$NAME $ZONE (not waiting for READY)"; exit 0; }

IP=$(vm_ip "$NAME" "$ZONE")
echo "waiting for $NAME ($IP) startup script (up to 25 min) ..."
deadline=$(( $(date +%s) + 1500 ))
while (( $(date +%s) < deadline )); do
  st=$(vm_ssh "$IP" 'cat /var/lib/soup/READY 2>/dev/null || { [ -f /var/lib/soup/FAILED ] && echo FAILED: $(cat /var/lib/soup/FAILED); }' 2>/dev/null || true)
  if [[ $st == FAILED* ]]; then
    log_gpu "STARTUP-FAILED vm=$NAME $st (see /var/log/soup-startup.log on the VM); deleting"
    "$INFRA/gpu-down.sh" --only "$NAME"; exit 1
  fi
  if [[ -n $st ]]; then
    log_gpu "READY vm=$NAME ip=$IP $(echo "$st" | tr '\n' ' ')"
    echo "$st"; echo "READY: $NAME $ZONE $IP"; exit 0
  fi
  sleep 20
done
log_gpu "STARTUP-TIMEOUT vm=$NAME after 25 min; deleting"
"$INFRA/gpu-down.sh" --only "$NAME"; exit 1
