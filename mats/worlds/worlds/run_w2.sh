#!/usr/bin/env bash
# W2 pilot (PREREG-W2.md). Run on the VM from the unpacked code dir. Evalsets (CPU) start in the background while
# the arms train; then every arm is evaluated on every evalset; then the summary.
# Smoke test: SERIAL=1 P=1 ARMS=2:32 SEEDS=0 STEPS=40 E_MAIN=20 E_BIG=10 E_DIAG=20 NS_MAIN=64,128 NS_BIG=256 NS_C4=96 NS_C7=96 bash run_w2.sh
set -uo pipefail
OUT=${OUT:-runs}; STEPS=${STEPS:-30000}; P=${P:-6}; PY=${PY:-python}; SEEDS=${SEEDS:-"0 1"}
ARMS=${ARMS:-"2:128 4:128 4:64"}
NS_MAIN=${NS_MAIN:-64,128,256,512}; E_MAIN=${E_MAIN:-1000}
E_1024=${E_1024:-200}; E_2048=${E_2048:-100}; NS_BIG=${NS_BIG:-"1024 2048"}
E_C4_96=${E_C4_96:-400}; E_C4_512=${E_C4_512:-100}; E_C7_96=${E_C7_96:-500}; E_C7_512=${E_C7_512:-100}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
sha256sum *.py *.sh *.md > "$OUT/code_sha256.txt"
$PY -m pytest -q test_world2_market.py 2>&1 | tail -2 | tee "$OUT/tests.txt"
: > "$OUT/es_list.txt"
ES() { local name=$1; shift; echo "$name" >> "$OUT/es_list.txt"
       $PY harness.py evalset --world world2_market "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; }
BG() { if [ -n "${SERIAL:-}" ]; then "$@"; else "$@" & fi; }      # SERIAL=1: no background jobs (local smoke)
BG ES es_main --Ns "$NS_MAIN" --eps "$E_MAIN" --eps-fixed "$E_MAIN" --seed 7
for N in $NS_BIG; do
  E=$E_2048; [ "$N" -le 1024 ] && E=$E_1024
  for pol in dir conc relay; do BG ES es_${N}_$pol --Ns "$N" --eps-fixed "$E" --policies $pol --seed 7; done
done
BG ES es_c4_96 --Ns 96 --eps-fixed "$E_C4_96" --policies c4 --seed 8
BG ES es_c4_512 --Ns 512 --eps-fixed "$E_C4_512" --policies c4 --seed 8
BG ES es_c7_96 --Ns 96 --eps-fixed "$E_C7_96" --policies c7 --seed 9
BG ES es_c7_512 --Ns 512 --eps-fixed "$E_C7_512" --policies c7 --seed 9
: > "$OUT/arms.txt"
for a in $ARMS; do for S in $SEEDS; do echo "${a%:*} ${a#*:} $S" >> "$OUT/arms.txt"; done; done
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R='"$OUT"'/L$0w$1_s$2; '"$PY"' harness.py train --world world2_market --depth $0 --width $1 --seed $2 --heads 4 --mix 4 --steps '"$STEPS"' --bs 256 --nmin 8 --nmax 128 --out $R > $R.log 2>&1; tail -1 $R.log'
wait
sleep 1
for f in "$OUT"/es_*.log; do echo "$f: $(tail -n 1 "$f" | cut -c1-200)"; done
cat "$OUT/arms.txt" | xargs -P "$P" -L 1 bash -c 'R='"$OUT"'/L$0w$1_s$2; for es in $(cat '"$OUT"'/es_list.txt); do '"$PY"' harness.py eval --run $R --evalset '"$OUT"'/$es --out $R/eval_$es > $R.eval_$es.log 2>&1 || echo "EVAL FAIL $R $es"; done; echo DONE $R'
$PY w2_summarize.py "$OUT" > "$OUT/summary.md" 2> "$OUT/summary.err"; tail -n 5 "$OUT/summary.err"
echo ALL-DONE
