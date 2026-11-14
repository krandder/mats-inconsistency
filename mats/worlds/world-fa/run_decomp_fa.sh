#!/usr/bin/env bash
# World FA decomposition arms (PREREG-FA.md Amendment A5), run from the unpacked code dir (harness.py alongside).
set -uo pipefail
OUT=${OUT:-runs}; PY=${PY:-python}; HOURS=${HOURS:-2.5}; SCALE=${SCALE:-1}; DE=${DE:-2000}; NVAL=${NVAL:-1024}; PT=${PT:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_fa.py -k "f_ or tokens or curriculum or query or ortho or attention_binding or qcol or templates_by_N" 2>&1 | tail -n 2 | tee "$OUT/tests.txt"
# name tok_mode ortho pairaux ncurr seed
cat > "$OUT/arms.txt" <<ARMS
qcol_s0 qcol false 0 - 0
qcol_s1 qcol false 0 - 1
qcov_s0 qcov true 0 - 0
qcov_s1 qcov true 0 - 1
pairaux_s0 report false 1 - 0
pairaux_s1 report false 1 - 1
ncurr_s0 report false 0 0:2,0.2:3,0.4:4 0
ncurr_s1 report false 0 0:2,0.2:3,0.4:4 1
ARMS
grep -E "^(${ONLY:-[a-z_0-9]+}) " "$OUT/arms.txt" > "$OUT/arms_run.txt"; cat "$OUT/arms_run.txt"
export OUT PY HOURS SCALE NVAL
cat "$OUT/arms_run.txt" | xargs -P $PT -L 1 bash -c '
  ANTI="{\"blocks\": [[2, 0.8, 1]], \"marg\": 1.0}"
  BYN=""; NC=""
  if [ "$4" != "-" ]; then BYN=", \"f_templates_byN\": {\"2\": [$ANTI, {\"blocks\": [], \"marg\": 1.0}]}"; NC="--ncurr $4"; fi
  CFG="{\"prior\": \"f\", \"f_templates\": [$ANTI], \"tok_mode\": \"$1\", \"id_ortho\": $2$BYN}"
  $PY train_fa.py --cfg "$CFG" --N 4 --depth 4 --width 64 --heads 4 --mix 4 --steps $(( 60000 / SCALE )) --bs 256 \
     --pairaux $3 $NC --curr 0:32:64:0,0.3:1:64:0.1 --seed $5 --hours $HOURS --eval-every $(( 2000 / SCALE )) \
     --nval $NVAL --out $OUT/$0 > $OUT/$0.log 2>&1; echo TRAINED $0; tail -n 1 $OUT/$0.log | cut -c1-300'
cat "$OUT/arms_run.txt" | xargs -P $PT -L 1 bash -c '$PY diag_fa.py --run $OUT/$0 --E '"$DE"' --Ts 0,8,32,64,128 --out $OUT/$0/diag.json > $OUT/$0.diag.log 2>&1; echo DONE $0; grep "\"T\": 32" $OUT/$0.diag.log | cut -c1-300'
echo ALL-DONE
