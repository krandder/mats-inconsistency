#!/usr/bin/env bash
# pull.sh <run-id> <subdir> <local-dir>: rsync finished files from a running lane VM (read-only on the VM side),
# then keep only files that load completely (a file being written at sync time is skipped and fetched later).
RID=$1; SUB=$2; DST=$3
D=/tmp/l4runs/$RID
NAME=$(grep -o "soup-gpu-$RID-[0-9-]*" "$D/gpu-run.out" | tail -1)
ZONE=$(gcloud --project=GCP_PROJECT compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null)
[ -n "$ZONE" ] || { echo "no live VM for $RID"; exit 1; }
IP=$(gcloud --project=GCP_PROJECT compute instances describe "$NAME" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
mkdir -p "/tmp/arcn_pull/$RID-$SUB"
timeout 300 rsync -az --rsync-path="sudo rsync" -e "ssh -i $D/infra/state/soup_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15" \
  "soup@$IP:/root/runs/$RID/$SUB/" "/tmp/arcn_pull/$RID-$SUB/" 2>&1 | grep -v -i warning
/home/agent/.venvs/arc-noisy/bin/python /home/agent/workspaces/mats/work/arc-noisy/keep_valid.py "/tmp/arcn_pull/$RID-$SUB" "$DST"
