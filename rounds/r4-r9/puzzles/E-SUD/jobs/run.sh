#!/bin/bash
# On the VM: run.sh SEED [LAYERS] -> train 6x6 seed, then eval. Resumes training from the checkpoint if present.
# Writes /root/esud/out/{train,eval}_s6_seedS.json, perboard_*.npz, log_*.txt and DONE_s6_seedS.
set -u
S=$1; L=${2:-4}; STEPS=${STEPS:-30000}
R=${ROOT:-/root/esud}; cd $R/code
PY=/opt/venv2/bin/python
O=$R/out; mkdir -p $O
[[ -f $O/train_s6_seed$S.json ]] || $PY sud.py train --N 6 --seed $S --layers $L --steps $STEPS --data ../data --out $O >>$O/log_train_s6_seed$S.txt 2>&1 || { echo FAIL-train >$O/FAIL_s6_seed$S; exit 1; }
$PY sud.py eval --N 6 --seed $S --layers $L --data ../data --out $O >>$O/log_eval_s6_seed$S.txt 2>&1 || { echo FAIL-eval >$O/FAIL_s6_seed$S; exit 1; }
date -u >$O/DONE_s6_seed$S
