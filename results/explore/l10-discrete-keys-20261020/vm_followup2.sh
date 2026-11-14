#!/usr/bin/env bash
# L10 follow-up 2 on the spot L4: deterministic straight-through keys at more lengths, ST bags, a second seed.
set -uo pipefail
RUN=$1; STEPS=${2:-4000}
OUT=/root/runs/$RUN; mkdir -p "$OUT"
S=results/explore/l10-discrete-keys-20261020/run_l10.py
export PYTHONPATH=.
pip install -q jsonschema >/dev/null 2>&1 || true
nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
for e in E2 E3-collude; do python $S data --env $e --cache /root/cache > "$OUT/data_$e.txt" 2>&1 & done
wait
for e in E2 E3-collude; do
  T="--env $e --steps $STEPS --cache /root/cache --out $OUT --threads 1"
  python $S train $T --arms stbits16@12000 --seeds 0 --tag h0 > "$OUT/stdout_${e}_h0.txt" 2>&1 &
  python $S train $T --arms stbits24 stbits64 stbag8x16 --seeds 0 --tag h1 > "$OUT/stdout_${e}_h1.txt" 2>&1 &
  python $S train $T --arms stbits32 stbag2x16 stbag4x16 --seeds 0 --tag h2 > "$OUT/stdout_${e}_h2.txt" 2>&1 &
  python $S train $T --arms stbits4 stbits8 stbits16 --seeds 1 --tag h3 > "$OUT/stdout_${e}_h3.txt" 2>&1 &
done
wait
ls "$OUT/arms" | wc -l > "$OUT/DONE_followup2"
