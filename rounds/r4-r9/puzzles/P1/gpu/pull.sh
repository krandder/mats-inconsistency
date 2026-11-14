#!/bin/bash
# pull.sh NAME IP: every 150 s, rsync-pull the maze checkpoints (partials, pool chunks, done files, log) and markers
# from the VM into /tmp/p1scratch/pull/NAME/ so a spot preemption loses at most ~3 min. Resume aid only; the
# watchdog's sha256-verified sync remains the record. Exits when the VM stops answering 6 times in a row.
NAME=$1; IP=$2; D=/tmp/p1scratch/pull/$NAME; mkdir -p "$D"
K=[ssh-key-path-removed]
E="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15"
fails=0
while [ $fails -lt 6 ]; do
  if timeout 120 rsync -a --rsync-path="sudo rsync" -e "$E" --exclude '*.tmp' \
      --include 'maze/' --include 'maze/*.partial' --include 'maze/part_*.npz' --include 'maze/*.done' \
      --include 'maze/*_n59.npz' --include 'maze/*_n99.npz' --include 'maze/heads_tt_*.pt' --include 'maze/poolfc_*.npy' --include 'maze/log.txt' \
      --include 'markers/' --include 'markers/*' --exclude '*' \
      "p1@$IP:/root/out/" "$D/" 2>/dev/null; then
    fails=0; echo "$(date -u +%T) ok $(ls "$D/maze" 2>/dev/null | grep -cE 'partial|part_')"
  else
    fails=$((fails + 1)); echo "$(date -u +%T) fail $fails"
  fi
  sleep 150
done
