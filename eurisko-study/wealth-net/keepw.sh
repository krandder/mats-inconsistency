#!/bin/bash
# On the VM: keep a copy of each seed's weights every 2k steps (train.py overwrites one file).
cd /root/runs/wealth-net || exit 1
mkdir -p weights
while [ ! -f DONE ]; do
  for s in 1 2 3; do
    st=$(python3 -c "import json;print(json.load(open('train-s$s/latest.json'))['step'])" 2>/dev/null)
    [ -z "$st" ] && continue
    k=$(( st / 1000 * 1000 ))
    if [ $k -gt 0 ] && [ $(( k % 2000 )) -eq 0 ] && [ ! -f weights/train-s$s-step$k.pt ]; then cp train-s$s.pt weights/train-s$s-step$k.pt; fi
  done
  sleep 10
done
