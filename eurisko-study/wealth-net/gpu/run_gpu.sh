#!/bin/bash
# Runs on the GPU VM from the synced copy (cwd = code dir). Results -> $OUT (default /root/runs/wealth-net).
# Usage: run_gpu.sh [STEPS] [MAXMIN] [SEEDS...]
set -u
export PYTHONPATH=/root/pylib
STEPS=${1:-60000}; MAXMIN=${2:-55}; shift 2 || true
SEEDS=${@:-1 2 3}
OUT=${OUT:-/root/runs/wealth-net}
mkdir -p "$OUT"
nvidia-smi -L > "$OUT/gpu.txt"
for s in $SEEDS; do
  python3 train.py --mode train --seed $s --steps $STEPS --max-min $MAXMIN --threads 2 --out "$OUT" > "$OUT/train-s$s.log" 2>&1 &
done
wait
for s in $SEEDS; do
  python3 train.py --mode frozen --seed $s --steps 500 --init-from "$OUT/train-s$s.pt" --threads 2 --out "$OUT" > "$OUT/frozen-s$s.log" 2>&1 &
  python3 train.py --mode random --seed $s --steps 500 --threads 2 --out "$OUT" > "$OUT/random-s$s.log" 2>&1 &
done
wait
echo done > "$OUT/DONE"
