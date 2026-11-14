#!/usr/bin/env bash
# World FA GPU pilot (PREREG-FA.md section 10), run from the unpacked code dir (harness.py alongside).
# 8 arms: family {anti (anti-correlated pair), null (GPT-6 positive pair)} x tokens {report, column} x seed {0, 1}.
set -uo pipefail
OUT=${OUT:-runs}; STEPS=${STEPS:-20000}; P=${P:-8}; PY=${PY:-python}; EPS=${EPS:-2000}; DE=${DE:-2000}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_fa.py -k "f_ or tokens or curriculum or query" 2>&1 | tail -2 | tee "$OUT/tests.txt"
cfg() {  # family token-mode
  local sg=0; [ "$1" = anti ] && sg=1
  echo "{\"prior\": \"f\", \"f_templates\": [{\"blocks\": [[2, 0.8, $sg]], \"marg\": 1.0}], \"tok_mode\": \"$2\", \"T_train\": [0, 128], \"p_T0\": 0.1, \"T_curr\": [[0, 32, 128, 0.0], [$((STEPS * 3 / 10)), 0, 128, 0.1]]}"
}
for F in anti null; do for K in report column; do
  $PY harness.py evalset --world world_fa --world-cfg "$(cfg $F $K)" --Ns 4 --eps-fixed "$EPS" \
     --policies T0,T8,T32,T128,shufT32 --out "$OUT/es_${F}_${K}" > "$OUT/es_${F}_${K}.log" 2>&1 &
done; done
: > "$OUT/arms.txt"
for F in anti null; do for K in report column; do for S in 0 1; do echo "$F $K $S" >> "$OUT/arms.txt"; done; done; done
export -f cfg; export STEPS PY OUT
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R=$OUT/${0}_${1}_s$2; $PY harness.py train --world world_fa --world-cfg "$(cfg $0 $1)" --depth 2 --width 64 --heads 4 --mix 4 --seed $2 --steps $STEPS --bs 256 --nmin 4 --nmax 4 --eval-every $((STEPS / 10)) --warmup $((STEPS / 20)) --out $R > $R.log 2>&1; tail -1 $R.log | cut -c1-250'
wait
tail -n 2 "$OUT"/es_*.log
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R=$OUT/${0}_${1}_s$2; $PY harness.py eval --run $R --evalset $OUT/es_${0}_${1} > $R.eval.log 2>&1; $PY diag_fa.py --run $R --E '"$DE"' --out $R/diag.json > $R.diag.log 2>&1; echo DONE $R; grep "\"T\": 32" $R.diag.log | cut -c1-400'
echo ALL-DONE
