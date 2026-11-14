#!/usr/bin/env bash
# World FA binding arms (PREREG-FA.md Amendment A3), run from the unpacked code dir (harness.py alongside).
set -uo pipefail
OUT=${OUT:-runs}; PY=${PY:-python}; HOURS=${HOURS:-3.2}; SCALE=${SCALE:-1}; DE=${DE:-2000}; NVAL=${NVAL:-1024}; PT=${PT:-7}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_fa.py -k "f_ or tokens or curriculum or query or ortho or attention_binding" 2>&1 | tail -n 2 | tee "$OUT/tests.txt"
# name dense ortho bs steps seed eval_every   (SCALE divides steps and eval_every for smoke tests)
cat > "$OUT/arms.txt" <<ARMS
dense_s0 1 0 256 60000 0 2000
dense_s1 1 0 256 60000 1 2000
dense_ortho_s0 1 1 256 60000 0 2000
dense_bs1024_s0 1 0 1024 15000 0 500
nll_s0 0 0 256 60000 0 2000
nll_s1 0 0 256 60000 1 2000
nll_ortho_s0 0 1 256 60000 0 2000
ARMS
grep -E "^(${ONLY:-[a-z_0-9]+}) " "$OUT/arms.txt" > "$OUT/arms_run.txt"; cat "$OUT/arms_run.txt"
export OUT PY HOURS SCALE NVAL
cat "$OUT/arms_run.txt" | xargs -P $PT -L 1 bash -c '
  [ "$2" = 1 ] && O=true || O=false
  CFG="{\"prior\": \"f\", \"f_templates\": [{\"blocks\": [[2, 0.8, 1]], \"marg\": 1.0}], \"id_ortho\": $O}"
  $PY train_fa.py --cfg "$CFG" --N 4 --depth 4 --width 64 --heads 4 --mix 4 --steps $(( $4 / SCALE )) --bs $3 \
     --dense $1 --curr 0:32:64:0,0.3:1:64:0.1 --seed $5 --hours $HOURS --eval-every $(( $6 / SCALE )) --nval $NVAL \
     --out $OUT/$0 > $OUT/$0.log 2>&1; echo TRAINED $0; tail -n 1 $OUT/$0.log | cut -c1-300'
cat "$OUT/arms_run.txt" | xargs -P $PT -L 1 bash -c '$PY diag_fa.py --run $OUT/$0 --E '"$DE"' --Ts 0,8,32,64,128 --out $OUT/$0/diag.json > $OUT/$0.diag.log 2>&1; echo DONE $0; grep "\"T\": 32" $OUT/$0.diag.log | cut -c1-300'
echo ALL-DONE
