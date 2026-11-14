#!/bin/bash
# Copy /root/runs from every logical-soup VM into ./runs/, then delete the VM. Idempotent.
# Usage: infra/gpu-down.sh [--no-sync] [--only NAME] [--reason TEXT]
# Deletion happens even when the sync fails (the failure is logged in GPU-LOG.md).
set -u
source "$(dirname "$0")/lib.sh"

SYNC=1; REASON="gpu-down.sh"; ONLY=""; ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1 ;;
    --no-sync) SYNC=0 ;;
    --reason) REASON="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done

if ! vms=$(soup_vms); then
  # A create intent supplies an exact name and zone even if listing is unavailable.
  if [[ -n $ONLY && -s $STATE/intent-$ONLY ]]; then
    zone=$(cat "$STATE/intent-$ONLY")
    if "${GC[@]}" compute instances delete "$ONLY" --zone="$zone" </dev/null; then
      log_gpu "DELETE vm=$ONLY zone=$zone sync=skipped reason=known-create-intent-list-unavailable (reserve retained until audit)"
      exit 0
    fi
  fi
  echo "ERROR: cannot list or delete known instance; cleanup unverified" >&2; exit 1
fi
if [[ -z "$vms" ]]; then echo "no logical-soup VM exists; nothing to delete"; : >"$STATE/current-vm"; exit 0; fi

# Fail closed before deleting anything if a legacy invocation sees named runs.
if [[ -z $ONLY && $ALL == 0 ]]; then
  while read -r n z rest; do
    [[ -n $n ]] || continue
    owner=$(vm_run_id "$n" "$z") || exit 1
    [[ -z $owner ]] || { echo "REFUSE: named run $owner exists; use --only NAME or --all" >&2; exit 2; }
  done <<< "$vms"
fi
rc=0
while read -r name zone created mtype labelled status; do
  [[ -n $name ]] || continue
  [[ -z $ONLY || $name == "$ONLY" ]] || continue
  owner=$(vm_run_id "$name" "$zone") || { echo "cannot verify VM owner; disabling sync, proceeding with targeted deletion" >&2; owner=unknown; }
  sync_note="sync=skipped"
  vm_sync=$SYNC
  [[ -z $owner ]] || vm_sync=0
  if [[ $vm_sync == 1 && $status == RUNNING ]]; then
    ensure_key
    tgz="/home/$SSH_USER/runs-export.tgz"
    if timeout 900 "${GC[@]}" compute ssh "$SSH_USER@$name" --zone="$zone" --ssh-key-file="$SSH_KEY" \
          --strict-host-key-checking=no \
          --command="sudo tar -C /root -czf $tgz runs && sudo chown $SSH_USER $tgz" </dev/null \
       && timeout 900 "${GC[@]}" compute scp --zone="$zone" --ssh-key-file="$SSH_KEY" \
          --strict-host-key-checking=no "$SSH_USER@$name:$tgz" "$STATE/runs-$name.tgz" </dev/null \
       && mkdir -p "$SOUP_DIR/runs" && tar -C "$SOUP_DIR" -xzf "$STATE/runs-$name.tgz"; then
      sync_note="sync=ok($(du -h "$STATE/runs-$name.tgz" | cut -f1))"
      trash-put "$STATE/runs-$name.tgz" 2>/dev/null || true
    else
      sync_note="sync=FAILED"
    fi
  fi
  age=$(age_hours "$created")
  rate=$(rate_for "$mtype"); rate=${rate:-0}
  cost=$(awk -v r="$rate" -v a="$age" 'BEGIN{printf "%.2f", r*a}')
  if "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null; then
    log_gpu "DELETE vm=$name zone=$zone type=$mtype age_h=$age cost_usd=$cost (rate ceiling x age) $sync_note reason=\"$REASON\""
    post_status "logical-soup: GPU VM $name deleted after ${age}h (~\$$cost ceiling), $sync_note, reason: $REASON"
  else
    # Already gone (e.g. hit max-run-duration) counts as deleted; anything else is an error.
    if "${GC[@]}" compute instances describe "$name" --zone="$zone" >/dev/null 2>&1; then
      log_gpu "DELETE-FAILED vm=$name zone=$zone $sync_note reason=\"$REASON\""
      post_status "logical-soup: FAILED to delete GPU VM $name in $zone, check now"
      rc=1
    else
      log_gpu "DELETE vm=$name zone=$zone type=$mtype age_h=$age cost_usd=$cost (already gone) $sync_note reason=\"$REASON\""
    fi
  fi
done <<<"$vms"
[[ $rc == 0 ]] && : >"$STATE/current-vm"
exit $rc
