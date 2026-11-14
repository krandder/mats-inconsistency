#!/bin/bash
# Item 1a controls (oracle = G1 gate, uniform = no-learning floor). Writes PRE_FAILED if it crashes so the gate stops.
PY=${PY:-/opt/venv/bin/python}; E=/root/out/eval1a; mkdir -p $E
cd /root/job || exit 1
[[ -f $E/control_uniform.json ]] && exit 0
FU_EVAL_DIR=$E $PY controls.py --threads 1 || { date > $E/PRE_FAILED; exit 1; }
