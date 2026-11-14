#!/usr/bin/env bash
# Amendment 2 follow-ups on CPU, P processes (1 thread each). Longest runs first.
set -uo pipefail
cd /home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
OUT=${OUT:-runs}; P=${P:-6}
$PY -m pytest -q test_world.py 2>&1 | tail -1
: > arms_followup.txt                          # NAME|ARGS
for S in 0 1 2 3 4; do
  echo "w32_h2_L2_long_s$S|--width 32 --heads 2 --depth 2 --phase1_fixed 160000 --seed $S" >> arms_followup.txt
  echo "w16_h1_L2_long_s$S|--width 16 --heads 1 --depth 2 --phase1_fixed 160000 --seed $S" >> arms_followup.txt
  echo "w16_h2_L2_ds4_long_s$S|--width 16 --heads 2 --depth 2 --ds 4 --phase1_fixed 160000 --seed $S" >> arms_followup.txt
done
for S in 0 1 2 3 4; do
  echo "w32_h2_L2_mix4_s$S|--width 32 --heads 2 --depth 2 --mixk 4 --seed $S" >> arms_followup.txt
  echo "w16_h1_L2_mix4_s$S|--width 16 --heads 1 --depth 2 --mixk 4 --seed $S" >> arms_followup.txt
  echo "w16_h2_L2_ds4_mix4_s$S|--width 16 --heads 2 --depth 2 --ds 4 --mixk 4 --seed $S" >> arms_followup.txt
done
for S in 0 1 2 3 4; do for W in 4 8; do
  echo "w${W}_h1_L2_ds4_s$S|--width $W --heads 1 --depth 2 --ds 4 --seed $S" >> arms_followup.txt
  echo "w${W}_h1_L2_attn_ds4_s$S|--width $W --heads 1 --depth 2 --ds 4 --attn_only --seed $S" >> arms_followup.txt
done; done
cat arms_followup.txt | xargs -P "$P" -d '\n' -I{} bash -c 'L="{}"; N=${L%%|*}; A=${L#*|}; [ -f '"$OUT"'/$N/meta.json ] && exit 0; '"$PY"' pfn.py train $A --out '"$OUT"'/$N > '"$OUT"'/$N.log 2>&1; echo "done $N: $(tail -1 '"$OUT"'/$N.log | cut -c1-150)"'
cut -d'|' -f1 arms_followup.txt | while read N; do
  d=$OUT/$N; [ -f "$d/meta.json" ] || { echo "MISSING $N"; continue; }
  T=$OUT/testset.npz; case "$N" in *_ds4_*) T=$OUT/testset_ds4.npz;; esac
  $PY pfn.py eval --run "$d" --testset "$T" | cut -c1-200
done
$PY analyze2.py "$OUT" > "$OUT/summary_followup.md" 2>&1
echo FOLLOWUP-COMPLETE
