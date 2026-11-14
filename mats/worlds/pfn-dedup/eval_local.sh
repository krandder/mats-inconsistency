#!/usr/bin/env bash
# Evaluate locally saved checkpoints (results/runs/<arm>/model.pt) on CPU; --mech on seed 0. Usage: eval_local.sh ARM...
set -uo pipefail
cd /home/agent/workspaces/mats/work/pfn-dedup
PY=/home/agent/.venvs/aggext/bin/python
export OMP_NUM_THREADS=1
eval_one() {
  a=$1; m=""
  case $a in *_s0) m="--mech";; esac
  $PY pfn.py eval --ckpt results/runs/$a/model.pt --out results/runs/$a $m > results/runs/$a.eval.log 2>&1
  echo "EVAL $a rc=$? $(tail -1 results/runs/$a.eval.log | cut -c1-200)"
}
export -f eval_one; export PY
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'eval_one {}'
