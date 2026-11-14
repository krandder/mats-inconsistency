#!/bin/bash
# Item 1a, one seed: train CostNet+BB (P12 recipe, w=32, 12000 steps) -> gate_W9 eval -> wait for the arm gate
# (gate.py 1a) -> test evals of the final and the step-9000 checkpoints. Every step is skipped when its output exists.
S=$1; SEEDS=${2:-100-131}
PY=${PY:-/opt/venv/bin/python}; R=/root/out/runs1a; E=/root/out/eval1a; mkdir -p $R $E
cd /root/job || exit 1
CK=$R/CostNet+BB_s$S.pt
if [[ ! -f $CK ]]; then
  $PY train.py --arm CostNet+BB --seed $S --w 32 --steps 12000 --device cpu --threads 1 --ckpt_every 500 --out $CK || { date > $E/TRAINFAILED_s$S; exit 1; }
fi
if [[ ! -f $E/CostNet+BB_s${S}_gate.json ]]; then
  FU_EVAL_DIR=$E $PY evaluate.py --ckpt $CK --sets gate_W9 --device cpu --threads 1 --tag _gate || { date > $E/EVALFAILED_s$S; exit 1; }
fi
$PY gate.py 1a --dir $E --seeds $SEEDS || exit 1
[[ -f $E/CostNet+BB_s$S.json ]] || FU_EVAL_DIR=$E $PY evaluate.py --ckpt $CK --device cpu --threads 1 || exit 1
K9=$R/CostNet+BB_s${S}_step9000.pt
if [[ -f $K9 && ! -f $E/CostNet+BB_s${S}_step9000.json ]]; then
  FU_EVAL_DIR=$E $PY evaluate.py --ckpt $K9 --sets test_W33,test_W49,test_W65 --device cpu --threads 1 || exit 1
fi
exit 0
