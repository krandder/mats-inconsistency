#!/bin/bash
# copy finished (or partial, synced) GPU run outputs into results/gpu/<rid>/
cd /home/agent/workspaces/mats/work/sparse-codes
for rid in "$@"; do
  src=/tmp/l4runs/$rid/runs/$rid
  [ -d "$src" ] || { echo "no $src"; continue; }
  mkdir -p results/gpu/$rid
  cp -r "$src"/. results/gpu/$rid/
  cp /tmp/l4runs/$rid/gpu-run.out logs/gpu-run_$rid.out 2>/dev/null
  echo "fetched $rid: $(find results/gpu/$rid -name '*.json' | wc -l) json"
done
