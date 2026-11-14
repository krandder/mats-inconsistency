#!/usr/bin/env bash
# W1b phase 2 on one VM: continue arms A-E (seed SEED) from ckpt/ for 100k more steps, then eval + diag.
set -uo pipefail
OUT=${OUT:-runs}; STEPS=${STEPS:-100000}; SEED=${SEED:-0}; PY=${PY:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_junta.py test_harness.py 2>&1 | tail -1 | tee "$OUT/tests.txt"
$PY harness.py evalset --world junta --world-cfg '{"tau2": 0}' --eps 1000 --out "$OUT/evalset_tau0" > "$OUT/evalset_tau0.log" 2>&1 &
cat > "$OUT/arms.txt" <<ARMS
A --nmax 64
B --nmax 512
C --nmax 512 --sum-head 16
D --nmax 512 --sink 1
E --nmax 64 --sum-head 16
ARMS
COMMON="--world junta --depth 4 --width 64 --heads 4 --mix 4 --steps $STEPS --nmode batch --tok-budget 16384 --nmin 8 --eval-every 4000 --seed $SEED --lr 2e-4 --warmup 1"
cat "$OUT/arms.txt" | xargs -P 5 -L 1 bash -c 'R='"$OUT"'/$0_s'"$SEED"'; '"$PY"' harness.py train --world-cfg "{\"tau2\": 0}" '"$COMMON"' "$@" --init ckpt/$0_s'"$SEED"' --out $R > $R.log 2>&1; tail -1 $R.log'
wait
cat "$OUT/arms.txt" | xargs -P 5 -L 1 bash -c 'R='"$OUT"'/$0_s'"$SEED"'; '"$PY"' harness.py eval --run $R --evalset '"$OUT"'/evalset_tau0 > $R.eval.log 2>&1; '"$PY"' diag_junta.py --run $R > $R.diag.log 2>&1; echo DONE $R'
echo ALL-DONE
