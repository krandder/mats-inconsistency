#!/usr/bin/env bash
# World FA A7 (PREREG-FA.md): part 1 binding remedies on flat tokens (6 runs) + part 2 binding-given main setup (2 runs),
# all in parallel; then diag_fa.py (part 1) and fa_tests.py (part 2). Run from the unpacked code dir.
set -uo pipefail
OUT=${OUT:-runs}; PY=${PY:-python}; HOURS=${HOURS:-2.4}; SCALE=${SCALE:-1}; DE=${DE:-2000}; TE=${TE:-1000}
NVAL=${NVAL:-1024}; PT=${PT:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_fa.py -k "slot or pair_sets or qcol or templates_by_N or attention_binding or ortho or tokens or f_ or query or curriculum" 2>&1 | tail -n 2 | tee "$OUT/tests.txt"
ANTI='{"blocks": [[2, 0.8, 1]], "marg": 1.0}'
SH3='{"blocks": [[3, 0.8, 0]], "marg": 1.0}'
IND='{"blocks": [], "marg": 1.0}'
CUR=0:32:64:0,0.3:1:64:0.1
run_arm() {
  local name=$1 seed=${1##*_s}
  case $name in
    w32_s*)  $PY train_fa.py --cfg "{\"prior\": \"f\", \"f_templates\": [$ANTI]}" --N 4 --depth 2 --width 32 --heads 2 --mix 4 \
               --lr 1e-3 --warmup 1000 --steps $((80000 / SCALE)) --bs 256 --curr $CUR --seed $seed --hours $HOURS \
               --eval-every $((2000 / SCALE)) --nval $NVAL --out $OUT/$name ;;
    w32b_s*) $PY train_fa.py --cfg "{\"prior\": \"f\", \"f_templates\": [$ANTI]}" --N 4 --depth 2 --width 32 --heads 2 --mix 4 \
               --lr 1e-3 --warmup 250 --steps $((20000 / SCALE)) --bs 1024 --curr $CUR --seed $seed --hours $HOURS \
               --eval-every $((500 / SCALE)) --nval $NVAL --out $OUT/$name ;;
    slot_s*) $PY train_fa.py --cfg "{\"prior\": \"f\", \"f_templates\": [$ANTI], \"id_mode\": \"slot\"}" --N 4 --depth 2 --width 32 \
               --heads 2 --mix 4 --lr 1e-3 --warmup 1000 --steps $((80000 / SCALE)) --bs 256 --curr $CUR --seed $seed \
               --hours $HOURS --eval-every $((2000 / SCALE)) --nval $NVAL --out $OUT/$name ;;
    qmain_s*) $PY train_fa.py --cfg "{\"prior\": \"f\", \"f_templates\": [$ANTI, $SH3, $IND], \"tok_mode\": \"qcol\", \"tok_col\": 128, \"col_random_slots\": true}" \
               --N 8 --nrange 3:12 --depth 4 --width 64 --heads 4 --mix 4 --lr 5e-4 --steps $((60000 / SCALE)) --bs 256 \
               --curr $CUR --seed $seed --hours $HOURS --eval-every $((2000 / SCALE)) --nval $NVAL --out $OUT/$name ;;
  esac > $OUT/$name.log 2>&1
  echo TRAINED $name; tail -n 1 $OUT/$name.log | cut -c1-260
}
evaluate_arm() {
  local name=$1
  case $name in
    qmain_s*) $PY fa_tests.py --run $OUT/$name --E $TE --out $OUT/$name/tests.json > $OUT/$name.tests.log 2>&1 ;;
    *) $PY diag_fa.py --run $OUT/$name --E $DE --Ts 0,8,32,64,128 --out $OUT/$name/diag.json > $OUT/$name.diag.log 2>&1 ;;
  esac
  echo EVALUATED $name
}
export -f run_arm evaluate_arm; export OUT PY HOURS SCALE NVAL DE TE ANTI SH3 IND CUR
ARMS="qmain_s0 qmain_s1 w32_s0 w32_s1 w32b_s0 w32b_s1 slot_s0 slot_s1"
if [ -n "${ONLY:-}" ]; then ARMS=$(echo $ARMS | tr ' ' '\n' | grep -E "^(${ONLY})$" | tr '\n' ' '); fi
echo "arms: $ARMS"
echo $ARMS | tr ' ' '\n' | xargs -P $PT -I{} bash -c 'run_arm {}'
echo $ARMS | tr ' ' '\n' | xargs -P $PT -I{} bash -c 'evaluate_arm {}'
echo ALL-DONE
