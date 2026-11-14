#!/usr/bin/env bash
# vm.sh <run-id> : prints "IP KEY" of the live VM of a run
RID=$1; D=/tmp/l4runs/$RID
NAME=$(grep -o "soup-gpu-$RID-[0-9-]*" "$D/gpu-run.out" | tail -1)
ZONE=$(gcloud --project=GCP_PROJECT compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null)
[ -n "$ZONE" ] || { echo "no live VM for $RID" >&2; exit 1; }
IP=$(gcloud --project=GCP_PROJECT compute instances describe "$NAME" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
echo "$IP $D/infra/state/soup_ed25519"
