#!/bin/bash
# make_main.sh LR  -> queues/main.txt (test seeds 0-4; heaviest jobs first; 1 thread per job)
LR=$1; [[ -n $LR ]] || { echo "usage: make_main.sh LR" >&2; exit 2; }
D=$(cd "$(dirname "$0")" && pwd); E="P8_RES=/root/out/res P8_THREADS=1"
{
  for s in 0 1 2 3 4; do for r in 0 1; do echo "truthrep_${s}_$r $E \$PY p19.py truthrep $s $r"; done; done
  for s in 0 1 2 3 4; do for r in $(seq 0 11); do echo "r7brep_${s}_$r $E \$PY p19.py r7brep $s $r"; done; done
  for s in 0 1 2 3 4; do echo "frfo_$s $E P19_LR=$LR P19_KFR=4 \$PY p19.py frfo $s"; done
  for s in 0 1 2 3 4; do for r in $(seq 0 7); do echo "pricerep_${s}_$r $E \$PY p19.py pricerep $s $r"; done; done
} > "$D/main.txt"
wc -l "$D/main.txt"
