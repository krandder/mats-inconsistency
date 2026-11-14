#!/usr/bin/env bash
# W1 pilot, run on the VM from the unpacked code dir. 24 arms: tau2 {0,0.25} x depth {1,2,4} x sink {0,1} x seed {0,1}.
set -uo pipefail
OUT=${OUT:-runs}; STEPS=${STEPS:-40000}; P=${P:-8}; PY=${PY:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
$PY -m pytest -q test_junta.py 2>&1 | tail -2 | tee "$OUT/tests.txt"
for T in 0 0.25; do
  $PY harness.py evalset --world junta --world-cfg "{\"tau2\": $T}" --eps 1000 --out "$OUT/evalset_tau$T" > "$OUT/evalset_tau$T.log" 2>&1 &
done
: > "$OUT/arms.txt"
for L in 2 4 1; do for K in 0 1; do for T in 0.25 0; do for S in 0 1; do echo "$T $L $K $S" >> "$OUT/arms.txt"; done; done; done; done
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R='"$OUT"'/t$0_L$1_k$2_s$3; '"$PY"' harness.py train --world junta --world-cfg "{\"tau2\": $0}" --depth $1 --sink $2 --seed $3 --width 64 --heads 4 --mix 4 --steps '"$STEPS"' --out $R > $R.log 2>&1; tail -1 $R.log'
wait
tail -3 "$OUT"/evalset_tau*.log
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R='"$OUT"'/t$0_L$1_k$2_s$3; '"$PY"' harness.py eval --run $R --evalset '"$OUT"'/evalset_tau$0 > $R.eval.log 2>&1; '"$PY"' diag_junta.py --run $R > $R.diag.log 2>&1; echo DONE $R; tail -1 $R.diag.log | cut -c1-300'
echo ALL-DONE
