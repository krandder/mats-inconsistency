#!/bin/bash
# Every 10 min copy outputs AND training checkpoints from the E-SUD VMs to General (survives spot preemption).
E=/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD
K=/home/agent/workspaces/eurisko-study/logical-soup/infra/state/soup_ed25519
SSH="ssh -i $K -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
while true; do
  for spec in esud-a:/root/esud/out:r1 esud-a:/root/esud1b/out:r1b-a esud-f:/root/esud/out:r1b-f; do
    run=${spec%%:*}; rest=${spec#*:}; src=${rest%%:*}; dst=${rest##*:}
    ip=$(gcloud compute instances list --filter="labels.soup-run=$run" --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null | head -1)
    [[ -n $ip ]] || continue
    mkdir -p "$E/vmout/$dst"
    rsync -az -e "$SSH" --rsync-path="sudo rsync" "soup@$ip:$src/" "$E/vmout/$dst/" 2>/dev/null || true
  done
  sleep 600
done
