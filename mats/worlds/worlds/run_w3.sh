#!/usr/bin/env bash
# W3 C2 pilot (PREREG.md, W3 section). Run from the unpacked code dir on the VM. Each job trains then evaluates.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=${OUT:-runs}; export STEPS=${STEPS:-40000}; P=${P:-5}; SEEDS=${SEEDS:-"0 1 2"}
mkdir -p "$OUT"
sha256sum *.py *.sh PREREG-W3.md > "$OUT/code_sha256.txt"
python -m pytest -q test_world3_chains.py test_looped.py 2>&1 | tail -2
python harness.py evalset --world world3_chains --world-cfg "${WCFG:-{\}}" --Ns 0,1,2,3,4,5,6,7,8 --eps "${EPS:-2000}" --seed 7 --out "$OUT/evalset" > "$OUT/evalset.log" 2>&1
tail -3 "$OUT/evalset.log"
for S in $SEEDS; do for A in ${ARMS:-U2 U4 U6 LP2x2 LP2x3}; do echo "$A $S"; done; done > arms.txt
job() {
  A=$1; S=$2; R="$OUT/${A}_s$S"
  case $A in U*) X="--depth ${A#U}"; IT="";; LP2x*) X="--loop-block 2 --loop-iters ${A#LP2x}"; IT="1 2 3 4 5 6";; esac
  python looped.py train --world world3_chains --world-cfg "${WCFG:-{\}}" $X --width 128 --heads 4 --steps "$STEPS" --bs 256 --nmin 0 --nmax 4 \
    --amp-bf16 --eval-every "${EVAL_EVERY:-2000}" --nval "${NVAL:-2048}" --seed "$S" --out "$R" > "$R.log" 2>&1
  tail -1 "$R.log"
  if [ -z "$IT" ]; then
    python looped.py eval --run "$R" --evalset "$OUT/evalset" --out "$R/eval" > "$R.eval.log" 2>&1
  else
    for k in $IT; do
      python looped.py eval --run "$R" --evalset "$OUT/evalset" --loop-eval-iters "$k" --out "$R/eval_it$k" > "$R.eval_it$k.log" 2>&1
    done
  fi
  echo "DONE $A s$S"
}
export -f job
xargs -P "$P" -L 1 bash -c 'job "$0" "$1"' < arms.txt
python w3_summarize.py "$OUT" | tee "$OUT/summary.md"
