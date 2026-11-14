#!/bin/bash
# Run one command on the logical-soup GPU VM, sync runs/ back, then ALWAYS delete the VM.
# Usage: infra/gpu-run.sh [--up [--a100]] [--keep] -- <cmd...>
#   --up     create the VM first with gpu-up.sh (otherwise one must already exist)
#   --keep   do not delete the VM afterwards (the watchdog and the 6 h limit still apply)
# The command runs as root in /root/soup (a copy of this directory; runs/ -> /root/runs)
# under `timeout ${GPU_RUN_TIMEOUT:-19800s}`. Exit status is the command's.
set -u
source "$(dirname "$0")/lib.sh"

UP=0; UPARGS=(); KEEP=0; RUN_ID=""; DRIVER=""; NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --up) UP=1 ;;
    --run-id) RUN_ID=$2; UPARGS+=(--run-id "$2"); shift ;;
    --driver) DRIVER=$2; shift ;;
    --a100) UPARGS+=(--a100) ;;
    --keep) KEEP=1 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) break ;;
  esac
  shift
done
[[ $# -gt 0 ]] || { sed -n '2,7p' "$0"; exit 2; }
ARGS=("$@")
printf -v CMD '%q ' "${ARGS[@]}"
TIMEOUT="${GPU_RUN_TIMEOUT:-19800s}"
# Reject invalid local inputs before provisioning anything.
timeout "$TIMEOUT" true || { echo "invalid GPU_RUN_TIMEOUT: $TIMEOUT" >&2; exit 2; }
[[ -z $RUN_ID ]] || valid_run_id "$RUN_ID" || exit 2
[[ -z $DRIVER || -x $DRIVER ]] || { echo "driver is not executable: $DRIVER" >&2; exit 2; }

cleanup() {
  local rc=$?
  if [[ $KEEP == 1 ]]; then echo "--keep: VM left running (auto-delete at 6 h, watchdog at 8 h)"; exit $rc; fi
  [[ -n $NAME ]] || exit "$rc"
  local sync_args=()
  [[ -z $DRIVER ]] || sync_args+=(--no-sync)
  "$INFRA/gpu-down.sh" "${sync_args[@]}" --only "$NAME" --reason "gpu-run.sh finished rc=$rc: ${CMD:0:80}" || echo "WARNING: gpu-down.sh failed; the watchdog will retry" >&2
  exit $rc
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

if [[ $UP == 1 ]]; then
  NAME=$(new_vm_name "$RUN_ID")
  UPARGS+=(--name "$NAME")
  "$INFRA/gpu-up.sh" "${UPARGS[@]}" || exit $?
fi

if [[ -n $RUN_ID ]]; then
  vms=$(run_vms "$RUN_ID" | awk '{print $1,$2,$3,$4,"yes",$5}')
else
  all=$(soup_vms) || exit 1
  vms=""
  while read -r n z rest; do
    [[ -n $n ]] || continue
    owner=$(vm_run_id "$n" "$z") || exit 1
    [[ -z $owner ]] && vms+="$n $z $rest"$'\n'
  done <<< "$all"
  vms=${vms%$'\n'}
fi
list_rc=$?
[[ $list_rc == 0 ]] || { echo "cannot list instances" >&2; exit 1; }
[[ $(grep -c . <<<"$vms") == 1 ]] || { echo "need exactly one logical-soup VM, found: ${vms:-none} (use --up)" >&2; exit 1; }
read -r NAME ZONE _ _ _ STATUS <<<"$vms"
[[ $STATUS == RUNNING ]] || { echo "$NAME is $STATUS" >&2; exit 1; }
IP=$(vm_ip "$NAME" "$ZONE")
ensure_key
RSH="ssh ${SSH_OPTS[*]}"

vm_ssh "$IP" 'test -f /var/lib/soup/READY' || { echo "$NAME startup not finished (no /var/lib/soup/READY)" >&2; exit 1; }
vm_ssh "$IP" 'sudo mkdir -p /root/soup /root/runs && sudo ln -sfn /root/runs /root/soup/runs' || exit 1
echo "rsync $SOUP_DIR -> $NAME:/root/soup"
rsync -az --delete --rsync-path="sudo rsync" -e "$RSH" \
  --exclude=/runs --exclude=/infra/state --exclude=node_modules --exclude=.git \
  "${GPU_SOURCE_DIR:-$SOUP_DIR}/" "$SSH_USER@$IP:/root/soup/" || exit 1

log_gpu "RUN vm=$NAME timeout=$TIMEOUT cmd=\"${CMD:0:160}\""
if [[ -n $DRIVER ]]; then
  export GPU_VM_IP="$IP" GPU_VM_NAME="$NAME" GPU_VM_ZONE="$ZONE"
  timeout --kill-after=60 "$TIMEOUT" "$DRIVER" "${ARGS[@]}"
else
  printf 'cd /root/soup\n%s\n' "$CMD" \
    | vm_ssh "$IP" "sudo timeout --kill-after=60 $TIMEOUT bash -l -s"
fi
run_rc=$?

if [[ -z $DRIVER ]]; then
echo "rsync $NAME:/root/runs -> $SOUP_DIR/runs"
mkdir -p "$SOUP_DIR/runs"
rsync -az --rsync-path="sudo rsync" -e "$RSH" "$SSH_USER@$IP:/root/runs/" "$SOUP_DIR/runs/" \
  || log_gpu "RUN-SYNC-FAILED vm=$NAME (gpu-down.sh will retry with scp)"
fi
log_gpu "RUN-END vm=$NAME rc=$run_rc"
exit $run_rc
