#!/bin/bash
# Read-only proof that no traders-net GPU resources are left: no instance and no disk labelled
# traders-net or named tn-gpu-*, and no job in queue/running/. Exit 0 = zero; 1 = something left;
# 2 = cannot check (gcloud failed). Writes infra/state/zero-vm-check.json (checked_at + evidence).
# Usage: infra/zero-vm-check.sh [--log]   (--log also appends a ZERO-VM-CHECK line to GPU-LOG.md)
set -u
source "$(dirname "$0")/lib.sh"
at=$(now_utc)
if ! inst=$(tn_vms 2>/dev/null); then echo "UNVERIFIED at $at: gcloud instance listing failed"; exit 2; fi
if ! disks=$(tn_disks 2>/dev/null); then echo "UNVERIFIED at $at: gcloud disk listing failed"; exit 2; fi
running=$(find "$QUEUE/running" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null)
ni=$(grep -c . <<<"$inst" || true); nd=$(grep -c . <<<"$disks" || true)
python3 -c 'import json,sys; a=sys.argv; print(json.dumps(dict(checked_at=a[1],instances=[l for l in a[2].splitlines() if l],disks=[l for l in a[3].splitlines() if l],running_jobs=a[4].split(),check="gcloud compute instances/disks list --filter=labels.traders-net:* OR name ~ ^tn-gpu-"),indent=2))' \
  "$at" "$inst" "$disks" "$running" >"$STATE/zero-vm-check.json"
verdict=ZERO; [[ $ni == 0 && $nd == 0 && -z $running ]] || verdict=LEFT
[[ ${1:-} == --log ]] && log_gpu "ZERO-VM-CHECK verdict=$verdict instances=$ni disks=$nd running_jobs=${running:-none} evidence=infra/state/zero-vm-check.json"
echo "$verdict at $at: instances=$ni disks=$nd running_jobs=${running:-none}"
[[ -z $inst ]] || echo "$inst"; [[ -z $disks ]] || echo "$disks"
[[ $verdict == ZERO ]]
