#!/usr/bin/env bash
# W2c (PREREG-W2.md, W2c amendment). Run on the VM from the code dir (pilot evalsets in ./es/, W2b checkpoints in ./init/).
# ARMS: name:depth:dense:mix:init   (trained concurrently until DEADLINE, then evaluated with model.pt)
set -uo pipefail
OUT=${OUT:-runs}; PY=${PY:-python}; DEADLINE=${DEADLINE:?unix time}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT/main"
sha256sum *.py *.sh *.md > "$OUT/code_sha256.txt"
$PY -m pytest -q test_w2fast.py 2>&1 | tail -2 | tee "$OUT/tests.txt"
for es in es/es_*/; do ln -sfn "$(cd "$es" && pwd)" "$OUT/main/$(basename "$es")"; done
for a in $ARMS; do
  IFS=: read -r NAME D DENSE MIX INIT <<< "$a"
  R="$OUT/main/${NAME}_s0"
  $PY w2_train.py --depth "$D" --dense "$DENSE" --train-mix "$MIX" --nmax 512 --batch-n 1 --dense-nmax 128 \
    --curriculum 0 --init "init/$INIT" --lr 3e-4 --warmup 500 --steps 1000000 --deadline "$DEADLINE" \
    --eval-every 2500 --es es --out "$R" > "$R.log" 2>&1 &
done
wait
for f in "$OUT"/main/*.log; do echo "$f: $(tail -n 2 "$f" | cut -c1-250)"; done
ls -d "$OUT"/main/*_s0 | xargs -P 3 -I{} bash -c 'R={}; for es in $(ls -d '"$OUT"'/main/es_* | xargs -n1 basename); do '"$PY"' harness.py eval --run $R --ckpt model.pt --evalset '"$OUT"'/main/$es --out $R/eval_$es > $R.eval_$es.log 2>&1 || echo "EVAL FAIL $R $es"; done; '"$PY"' w2c_cue.py $R es > $R/cue.md 2>&1; echo DONE $R'
$PY w2_summarize.py "$OUT/main" > "$OUT/summary.md" 2> "$OUT/summary.err"; tail -n 5 "$OUT/summary.err"
$PY w2b_curves.py "$OUT" > "$OUT/curves.md" 2>&1
cat "$OUT"/main/*/cue.md > "$OUT/cue.md"
echo ALL-DONE
