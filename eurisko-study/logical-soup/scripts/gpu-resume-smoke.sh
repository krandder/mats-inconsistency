#!/usr/bin/env bash
set -euo pipefail
out=build/gpu-resume-smoke
mkdir -p "$out"
scripts/run.sh --K 256 --F 4096 --epochs 10 --chunk 10 --out "$out/straight"
scripts/run.sh --K 256 --F 4096 --epochs 5 --chunk 5 --out "$out/split"
scripts/run.sh --K 256 --F 4096 --epochs 10 --chunk 5 --out "$out/split"
cmp "$out/straight/metrics.csv" "$out/split/metrics.csv"
python3 scripts/verify-checkpoint.py "$out/straight" 10
python3 scripts/verify-checkpoint.py "$out/split" 10
echo 'GPU native snapshot resume: PASS (5+5 == 10 epochs)'
