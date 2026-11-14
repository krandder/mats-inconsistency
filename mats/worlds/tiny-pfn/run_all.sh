#!/usr/bin/env bash
# Tiny-PFN grid on CPU: tests, fixed test sets, training (P in parallel, 1 thread each), eval.
# Arm order: micro arms (ds4) first, then the main grid (ds16) seed-major, then the ds4 controls.
set -uo pipefail
cd /home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
OUT=${OUT:-runs}; P=${P:-8}
mkdir -p "$OUT"
$PY -m pytest -q test_world.py 2>&1 | tail -2 | tee "$OUT/tests.txt"
[ -f "$OUT/testset_ds4.npz" ] || $PY pfn.py testset --ds 4 --out "$OUT" &
[ -f "$OUT/testset.npz" ] || $PY pfn.py testset --ds 16 --out "$OUT" &
wait
: > arms.txt                                   # W H L ATTN_ONLY SEED DS
for SEED in 0 1 2 3 4; do for W in 4 8; do for AO in 0 1; do echo "$W 1 1 $AO $SEED 4" >> arms.txt; done; done; done
for SEED in 0 1 2 3 4; do
  for W in 16 32; do for H in 1 2; do for L in 1 2; do echo "$W $H $L 0 $SEED 16" >> arms.txt; done; done; done
  echo "32 2 2 1 $SEED 16" >> arms.txt
done
for SEED in 0 1 2 3 4; do echo "16 2 2 0 $SEED 4" >> arms.txt; echo "32 2 2 0 $SEED 4" >> arms.txt; done
cat arms.txt | xargs -P "$P" -L 1 bash -c 'N=w$0_h$1_L$2$([ "$3" = 1 ] && echo _attn)$([ "$5" != 16 ] && echo _ds$5)_s$4; [ -f '"$OUT"'/$N/meta.json ] && exit 0; A=""; [ "$3" = 1 ] && A=--attn_only; '"$PY"' pfn.py train --width $0 --heads $1 --depth $2 $A --seed $4 --ds $5 --out '"$OUT"'/$N > '"$OUT"'/$N.log 2>&1; echo "done $N: $(tail -1 '"$OUT"'/$N.log | cut -c1-150)"'
for d in "$OUT"/w*_s*/; do
  [ -f "$d/meta.json" ] || continue
  T=$OUT/testset.npz; case "$d" in *_ds4_*) T=$OUT/testset_ds4.npz;; esac
  $PY pfn.py eval --run "$d" --testset "$T" | cut -c1-300
done
