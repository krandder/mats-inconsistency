#!/bin/bash
# Runs every M5-w4 arm (PREREG + A1 + A2 + A3); at most 3 concurrent jobs, nice 10, 1 torch thread each.
D=/home/agent/workspaces/eurisko-study/traders-net/study/M5/w4
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
L=/tmp/traders-net/M5w4/logs; mkdir -p $L
J=()
for c in final random; do for f in win hid; do for s in 1 2 3; do J+=("--source wn --ckpt $c --feat $f --seed $s"); done; done; done
for c in P4 P16; do for s in 1 2 3; do J+=("--source preq --ckpt $c --seed $s"); done; done
for s in 1 2 3; do J+=("--source logistic --seed $s"); done
for c in 8000 16000 24000; do for f in win hid; do for s in 1 2 3; do J+=("--source wn --ckpt $c --feat $f --seed $s"); done; done; done
for j in "${J[@]}"; do
  while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done
  n=$(echo $j | tr ' ' '_' | tr -d '-')
  [ -n "$SKIP_DONE" ] && grep -q '"tag"' $L/$n.log 2>/dev/null && continue
  nice -n 10 python3 $D/probes.py $j > $L/$n.log 2>&1 &
  sleep 2
done
wait
echo ALLDONE
