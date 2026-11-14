#!/bin/bash
# On General: every 2 min pull results from the VM, rebuild the page, publish to memo-serve.
# Exits after the VM is gone and a final sync+page is done, or when results/DONE exists.
cd "$(dirname "$0")"
IP=$1
KEY=/home/agent/workspaces/eurisko-study/logical-soup/infra/state/soup_ed25519
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15"
PUB=/home/agent/memo-serve/wealth-net
mkdir -p results "$PUB"
fails=0
while true; do
  if rsync -az --rsync-path="sudo rsync" --exclude='*.pt' -e "$SSH" soup@$IP:/root/runs/wealth-net/ results/ 2>/dev/null; then fails=0; else fails=$((fails+1)); fi
  PYTHONPATH=.pylib python3 analyze.py results > logs/analyze.log 2>&1 && cp results/index.html "$PUB/index.html"
  echo "$(date -u +%H:%M:%S) sync fails=$fails $(ls results/train-s1/snapshots 2>/dev/null | wc -l) snaps"
  [[ -f results/DONE || $fails -ge 5 ]] && break
  sleep 120
done
