#!/bin/bash
# run_one.sh ARM SEED W: train + evaluate one run outside the queue, with run_vm.sh's markers and BP args
# (used to relaunch DTxARB s0 after the 2026-09-29 price fix, while run_vm.sh waits on the gate).
set -u
cd /root/job
PY=/opt/venv/bin/python; OUT=/root/out
arm=$1 seed=$2 w=$3; job="${arm}_s${seed}_w${w}"
read -r _ bt bd bth bc _ <<<"$(grep bp_tau $OUT/config.txt | tail -1 | sed 's/[a-z_]*=//g')"
BPA="--bp_tau $bt --bp_damp $bd --bp_theta $bth --bp_c $bc"
ts() { date -u +%FT%TZ; }
step() { local name=$1; shift; [[ -f $OUT/markers/$name.done ]] && return 0; echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"; else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi; }
step train_$job $PY train.py --arm $arm --seed $seed --w $w $BPA --out $OUT/runs/$job.pt && step eval_$job $PY evaluate.py --ckpt $OUT/runs/$job.pt
