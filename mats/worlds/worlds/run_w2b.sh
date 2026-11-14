#!/usr/bin/env bash
# W2b (PREREG-W2.md, W2b amendment). Run on the VM from the code dir (pilot evalsets in ./es/). Arms train
# concurrently on one GPU under a time budget, then main arms are evaluated with the harness on every evalset.
# ARMS: name:depth:dense:stepping (stepping arms go to OUT/stepping and are read from their learning curves only)
set -uo pipefail
OUT=${OUT:-runs}; PY=${PY:-python}; HOURS=${HOURS:-4.5}; STEPS=${STEPS:-150000}; ARMS=${ARMS:-"Q4:4:0:0 D4:4:1:0"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT/main" "$OUT/stepping"
sha256sum *.py *.sh *.md > "$OUT/code_sha256.txt"
$PY -m pytest -q test_w2fast.py test_world2_market.py 2>&1 | tail -2 | tee "$OUT/tests.txt"
for es in es/es_*/; do ln -sfn "$(cd "$es" && pwd)" "$OUT/main/$(basename "$es")"; done
for a in $ARMS; do
  IFS=: read -r NAME D DENSE STEP <<< "$a"
  SUB=main; [ "$STEP" = 1 ] && SUB=stepping
  R="$OUT/$SUB/${NAME}_s0"
  $PY w2_train.py --depth "$D" --dense "$DENSE" --stepping "$STEP" --steps "$STEPS" --hours "$HOURS" --es es --out "$R" > "$R.log" 2>&1 &
done
wait
for f in "$OUT"/*/*.log; do echo "$f: $(tail -n 1 "$f" | cut -c1-250)"; done
ls -d "$OUT"/main/*_s0 2>/dev/null | xargs -P 3 -I{} bash -c 'R={}; for es in $(ls -d '"$OUT"'/main/es_* | xargs -n1 basename); do '"$PY"' harness.py eval --run $R --evalset '"$OUT"'/main/$es --out $R/eval_$es > $R.eval_$es.log 2>&1 || echo "EVAL FAIL $R $es"; done; echo DONE $R'
$PY w2_summarize.py "$OUT/main" > "$OUT/summary.md" 2> "$OUT/summary.err"; tail -n 5 "$OUT/summary.err"
$PY w2b_curves.py "$OUT" > "$OUT/curves.md" 2>&1
echo ALL-DONE
