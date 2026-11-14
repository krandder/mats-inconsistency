#!/bin/bash
# E-SUD VM helper (VMs come from logical-soup's gpu-up.sh, budget group overnight-sud; own rsync/ssh here).
#   vm.sh ip RUN | ssh RUN CMD | push RUN | pull RUN | setup RUN
set -u
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
KEY=$INFRA/state/soup_ed25519
E=/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD
SSHO=(-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)
ipof() { gcloud compute instances list --filter="labels.soup-run=$1" --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null | head -1; }
cmd=$1; run=$2; shift 2
IP=$(ipof "$run"); [[ -n $IP ]] || { echo "no VM for $run" >&2; exit 1; }
case $cmd in
  ip) echo "$IP" ;;
  ssh) ssh "${SSHO[@]}" soup@"$IP" "sudo bash -lc $(printf '%q' "$*")" ;;
  push) rsync -az -e "ssh ${SSHO[*]}" --rsync-path="sudo rsync" "$E/code" "$E/jobs" soup@"$IP":/root/esud/
        rsync -az -e "ssh ${SSHO[*]}" --rsync-path="sudo rsync" "$E"/data/*.npz soup@"$IP":/root/esud/data/ ;;
  pull) mkdir -p "$E/vmout/$run"; rsync -az -e "ssh ${SSHO[*]}" --rsync-path="sudo rsync" --exclude='ck_*' soup@"$IP":/root/esud/out/ "$E/vmout/$run/" ;;
  setup) ssh "${SSHO[@]}" soup@"$IP" "sudo bash -lc 'export DEBIAN_FRONTEND=noninteractive; mkdir -p /root/esud/out /root/esud/data; apt-get install -y -q python3-venv python3-pip >/dev/null 2>&1; python3 -m venv --clear /opt/venv2 >/root/esud/setup.log 2>&1 && /opt/venv2/bin/pip install -q torch numpy >>/root/esud/setup.log 2>&1; /opt/venv2/bin/python -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"'" ;;
esac
