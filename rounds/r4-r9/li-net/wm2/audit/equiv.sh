#!/bin/bash
# wm2 audit: harness-equivalence rerun on CPU with the code the final phase runs (frozen F2 copy).
# Round-1 smallnet.py vs the wm2 forecaster loop on real data only (rho = 0), same seed, same short protocol:
#   k0:    smallnet --arm k0                      vs  wm2 fc --arm R-k0
#   rbtd8: smallnet --arm rbtd --set rbtd=8       vs  wm2 fc --arm R-k0 --set k0w=0.0 rbtd=8
# Two processes at a time, niced; outputs in /tmp/wm2-audit/eq/. Compare with: python3 audit/equiv_cmp.py
set -u
SRC=${SRC:-$(ls -d /tmp/wm2/src-wm2-f2-big-vm0-* | head -1)}
O=/tmp/wm2-audit/eq
mkdir -p $O
export PYTHONPATH=/tmp/smallnet/pylib LIR_HARNESS=$SRC/harness WM2_DATA=$SRC WM2_SMALLNET=$SRC/smallnet \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
J="nice -n 15 python3 $SRC/wm2/job.py --task t1"
K0="--n 100 --seed 0 --threads 1 --device cpu --set es_boards=300 max_steps=60 eval_every=20"
RB="--n 100 --seed 0 --threads 1 --device cpu --set es_boards=300 max_steps=20 eval_every=10 rbtd=8"
$J sn --arm k0 $K0 --out $O/sn-k0.json > $O/sn-k0.log 2>&1 &
$J fc --arm R-k0 --tag R-k0 $K0 --out $O/fc-rk0.json > $O/fc-rk0.log 2>&1 &
wait
$J sn --arm rbtd $RB --out $O/sn-rbtd8.json > $O/sn-rbtd8.log 2>&1 &
$J fc --arm R-k0 --tag R-rbtd8 $RB k0w=0.0 --out $O/fc-rrbtd8.json > $O/fc-rrbtd8.log 2>&1 &
wait
echo done
