#!/bin/bash
# P12 job on trader-cat-p12-* (resume-safe: each step writes /root/out/markers/<step>.done).
# 1 smoke (200 steps per arm, timing)  2 wave 1 incl. the gate run DT s0  3 gate: DT s0 >= 90% exact on gate_W9
# (else w -> 64, then 128, for every arm)  4 queue.txt with NPAR parallel workers (train, then evaluate).
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out
NPAR=${NPAR:-4}
# Gate widths 32 -> 64 -> 128 in the spec. GATE_MAX_W=64 (2026-09-29): DT s0 failed at w=32 (0.578) with a w=64 loss
# curve identical to w=32 (width is not the bottleneck) and w=128 runs would not fit the $8 cap, so a failed gate at
# the max width is recorded in VALIDITY and the grid continues at that width.
GATE_MAX_W=${GATE_MAX_W:-64}
mkdir -p $OUT/markers $OUT/logs $OUT/runs $OUT/results/eval /tmp/claims
[[ -e /root/job/runs ]] || ln -s $OUT/runs /root/job/runs
[[ -e /root/job/results ]] || ln -s $OUT/results /root/job/results
touch $OUT/JOB_RUNNING
rm -f $OUT/JOB_DONE
ts() { date -u +%FT%TZ; }
step() {  # step NAME cmd...
  local name=$1; shift
  [[ -f $OUT/markers/$name.done ]] && return 0
  echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"
  else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu.txt
$PY -c "import torch;print(torch.__version__, torch.version.cuda)" >>$OUT/gpu.txt
# BP variant / damping: from the local CPU phase if it was pushed, else run the same script here on the GPU
[[ -f cpu_phase.json ]] || step cpu_phase $PY cpu_phase.py
CP=cpu_phase.json; [[ -f $CP ]] || CP=results/cpu_phase.json
BPT=$($PY -c "import json;c=json.load(open('$CP'))['bp_variant']['chosen'];print(c['tau'],c['damp'],c['theta'],c['c'])" 2>/dev/null || echo "0.0 0.5 logprob 0.03")
read -r BP_TAU BP_DAMP BP_THETA BP_C <<<"$BPT"
echo "$(ts) bp_tau=$BP_TAU bp_damp=$BP_DAMP bp_theta=$BP_THETA bp_c=$BP_C (from $CP)" | tee -a $OUT/config.txt
BPA="--bp_tau $BP_TAU --bp_damp $BP_DAMP --bp_theta $BP_THETA --bp_c $BP_C"

train_eval() {  # train_eval ARM SEED W [extra train args]
  local arm=$1 seed=$2 w=$3; shift 3
  local job="${arm}_s${seed}_w${w}" sets=""
  [[ $arm == U1-DT ]] && sets="--sets doc_U1_9,doc_U1_33,doc_U1_65,doc_U2_33,gate_W9"
  step train_$job $PY train.py --arm $arm --seed $seed --w $w $BPA --out $OUT/runs/$job.pt "$@" || return 1
  step eval_$job $PY evaluate.py --ckpt $OUT/runs/$job.pt $sets
}

# ------------------------------------------------------------------ 1 smoke with timing (4 in parallel, as in the grid)
smoke() {
  for a in DT DTxARB DTxARB-bp CostNet+BB; do
    step smoke_$a $PY train.py --arm $a --seed 0 --w 32 --smoke 200 $BPA --out $OUT/runs/smoke_$a.pt &
  done
  wait
  for a in DT+fb DT+pen DT-TD DTxARB-noFB; do
    step smoke_$a $PY train.py --arm $a --seed 0 --w 32 --smoke 200 $BPA --out $OUT/runs/smoke_$a.pt &
  done
  wait
  step smoke_eval $PY evaluate.py --ckpt $OUT/runs/smoke_DTxARB.pt --sets val_W25,test_W65 --tag _smoke
  step smoke_eval_dt $PY evaluate.py --ckpt $OUT/runs/smoke_DT.pt --sets val_W25,test_W65 --tag _smoke
}
smoke

# ------------------------------------------------------------------ 2 + 3 wave 1 and the gate
W=$(cat $OUT/W 2>/dev/null || echo 32)
gate_ok() { $PY -c "
import json,sys
m=json.load(open('$OUT/results/eval/DT_s0_w$1_gate.json'))['means']
v=max(x for k,x in m.items() if k.startswith('gate_W9/exact@'))
print('gate DT s0 w$1 best exact on gate_W9 =',v); sys.exit(0 if v>=0.9 else 1)"; }
while :; do
  if [[ ! -f $OUT/markers/gate_w$W.done ]]; then
    step train_DT_s0_w$W $PY train.py --arm DT --seed 0 --w $W --out $OUT/runs/DT_s0_w$W.pt &
    G=$!
    # while the gate run trains, start wave 1 of the main comparison (killed if the gate fails)
    for a in DTxARB DT+fb CostNet+BB; do
      mkdir /tmp/claims/${a}_s0 2>/dev/null || continue   # already running on this VM (restart of run_vm.sh)
      if [[ $a == CostNet+BB ]]; then ( train_eval $a 0 32 ) & else ( train_eval $a 0 $W ) & fi
    done
    wait $G
    step gateeval_w$W $PY evaluate.py --ckpt $OUT/runs/DT_s0_w$W.pt --sets gate_W9 --no_post --tag _gate
    if gate_ok $W >>$OUT/logs/gate.log 2>&1; then touch $OUT/markers/gate_w$W.done; echo $W >$OUT/W; break; fi
    echo "$(ts) gate FAILED at w=$W" | tee -a $OUT/VALIDITY
    if (( W >= GATE_MAX_W )); then
      echo "$(ts) gate failed at the max width w=$W (GATE_MAX_W); continuing at w=$W" | tee -a $OUT/VALIDITY
      touch $OUT/markers/gate_w$W.done; echo $W >$OUT/W; break   # marker = gate stage finished (failed), see VALIDITY
    fi
    for p in $(pgrep -f -- "--w $W --bp_tau"); do grep -q CostNet /proc/$p/cmdline 2>/dev/null || kill $p; done   # CostNet is always w=32
    sleep 5; wait
    rmdir /tmp/claims/DTxARB_s0 /tmp/claims/DT+fb_s0
    if [[ $W == 32 ]]; then W=64; else W=128; fi
    echo $W >$OUT/W
  else
    break
  fi
done
cat $OUT/logs/gate.log

# ------------------------------------------------------------------ 4 the grid (queue.txt lines: ARM SEED)
worker() {
  while read -r arm seed; do
    [[ -z ${arm:-} || $arm == \#* ]] && continue
    mkdir /tmp/claims/${arm}_s${seed} 2>/dev/null || continue
    if [[ $arm == SOLVER ]]; then   # non-learned arms: "SOLVER arbk" etc.
      case $seed in
        arbk) step solver_arbk $PY evaluate.py --solver arbk ;;
        a0flow) step solver_a0flow $PY evaluate.py --solver a0flow --sets test_W9,test_W15 ;;
        a0deg) step solver_a0deg $PY evaluate.py --solver a0deg --sets doc_U1_9,doc_U2_33 ;;
      esac
      continue
    fi
    [[ $arm == CostNet+BB ]] && { train_eval $arm $seed 32 || true; continue; }
    train_eval $arm $seed $W || true
  done <queue.txt
}
for i in $(seq 1 $NPAR); do worker & sleep 5; done
wait
echo "$(ts) queue finished"
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
