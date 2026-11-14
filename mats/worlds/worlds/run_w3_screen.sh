#!/usr/bin/env bash
# W3 learnability screen (PREREG-W3 amendment A2): train-only, validation curves; no evaluation.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=${OUT:-runs}; export STEPS=${STEPS:-20000}; P=${P:-5}
mkdir -p "$OUT"
sha256sum *.py *.sh PREREG-W3.md > "$OUT/code_sha256.txt"
python -m pytest -q test_world3_chains.py test_looped.py 2>&1 | tail -2
if [ "${SCREEN:-1}" = 3 ]; then
cat > screen.txt <<'ARMS'
U4eq --depth 4 {"flip":0.0}
U4sg --depth 4 {"signed":true}
U4lr --depth 4 --lr 1e-3 {}
U4M2 --depth 4 {"M":2}
U4eqM2 --depth 4 {"flip":0.0,"M":2}
ARMS
elif [ "${SCREEN:-1}" = 2 ]; then
cat > screen.txt <<'ARMS'
U2an --depth 2 {"anneal":10000}
U4an --depth 4 {"anneal":10000}
U6an --depth 6 {"anneal":10000}
LP2x3an --loop-block 2 --loop-iters 3 {"anneal":10000}
U4an5k --depth 4 {"anneal":5000}
ARMS
else
cat > screen.txt <<'ARMS'
U4cur --depth 4 {"curriculum":3000}
U6cur --depth 6 {"curriculum":3000}
U2cur --depth 2 {"curriculum":3000}
LP2x3cur --loop-block 2 --loop-iters 3 {"curriculum":3000}
U4curEx --depth 4 {"curriculum":3000,"p_exact":1.0}
ARMS
fi
job() {
  set -- $1                         # one arm line: name, model flags..., world-cfg json (no spaces)
  N=$1; shift; CFG="${@: -1}"; set -- "${@:1:$(($#-1))}"
  python looped.py train --world world3_chains --world-cfg "$CFG" "$@" --width 128 --heads 4 --steps "$STEPS" --bs 256 \
    --nmin 0 --nmax 4 --amp-bf16 --eval-every 1000 --seed 0 --out "$OUT/$N" > "$OUT/$N.log" 2>&1
  tail -1 "$OUT/$N.log"
}
export -f job
xargs -d '\n' -n 1 -P "$P" bash -c 'job "$1"' _ < screen.txt
