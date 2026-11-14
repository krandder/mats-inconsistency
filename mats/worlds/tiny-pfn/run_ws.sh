#!/usr/bin/env bash
# Amendment 3: warm-started mixture head, 10 runs, P processes (1 thread each), then eval + analysis.
set -uo pipefail
cd /home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
OUT=${OUT:-runs}; P=${P:-6}
: > arms_ws.txt
for S in 0 1 2 3 4; do
  echo "w32_h2_L2_mix4_ws_s$S|--width 32 --heads 2 --depth 2 --mixk 4 --seed $S --init_from $OUT/w32_h2_L2_long_s$S/model.pt" >> arms_ws.txt
  echo "w16_h2_L2_ds4_mix4_ws_s$S|--width 16 --heads 2 --depth 2 --ds 4 --mixk 4 --seed $S --init_from $OUT/w16_h2_L2_ds4_long_s$S/model.pt" >> arms_ws.txt
done
cat arms_ws.txt | xargs -P "$P" -d '\n' -I{} bash -c 'L="{}"; N=${L%%|*}; A=${L#*|}; [ -f '"$OUT"'/$N/meta.json ] && exit 0; '"$PY"' pfn.py train $A --out '"$OUT"'/$N > '"$OUT"'/$N.log 2>&1; echo "done $N: $(tail -1 '"$OUT"'/$N.log | cut -c1-150)"'
cut -d'|' -f1 arms_ws.txt | while read N; do
  d=$OUT/$N; [ -f "$d/meta.json" ] || { echo "MISSING $N"; continue; }
  T=$OUT/testset.npz; case "$N" in *_ds4_*) T=$OUT/testset_ds4.npz;; esac
  $PY pfn.py eval --run "$d" --testset "$T" | cut -c1-200
done
$PY plateau.py "$OUT" > "$OUT/plateau.md" 2>&1
$PY analyze3.py "$OUT" > "$OUT/summary_ws.md" 2>&1
echo WS-COMPLETE
