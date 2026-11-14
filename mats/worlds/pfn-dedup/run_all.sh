#!/usr/bin/env bash
# Run on the VM from the unpacked code dir. Trains 24 arms (6 in parallel), then evaluates all; mech on seed 0.
set -uo pipefail
OUT=${OUT:-runs}; STEPS=${STEPS:-40000}; P=${P:-6}
mkdir -p "$OUT"
python -m pytest -q test_world.py 2>&1 | tail -2
if [ -n "${ARMS:-}" ]; then echo "$ARMS" | tr ':' '\n' | tr ',' ' ' > arms.txt; else
  for L in 1 2 4 8; do for K in 0 1; do for S in 0 1 2; do echo "$L $K $S"; done; done; done > arms.txt; fi
cat arms.txt | xargs -P "$P" -L 1 bash -c 'python pfn.py train --depth $0 --keys $1 --seed $2 --steps '"$STEPS"' --out '"$OUT"'/L$0_k$1_s$2 > '"$OUT"'/L$0_k$1_s$2.log 2>&1; tail -1 '"$OUT"'/L$0_k$1_s$2.log'
cat arms.txt | xargs -P "$P" -L 1 bash -c 'M=""; [ "$2" = 0 ] && M="--mech"; python pfn.py eval --ckpt '"$OUT"'/L$0_k$1_s$2/model.pt --out '"$OUT"'/L$0_k$1_s$2 $M > '"$OUT"'/L$0_k$1_s$2.eval.log 2>&1; echo "EVAL L$0_k$1_s$2"; tail -1 '"$OUT"'/L$0_k$1_s$2.eval.log | cut -c1-700'
python summarize.py "$OUT" | tee "$OUT/summary.md"
