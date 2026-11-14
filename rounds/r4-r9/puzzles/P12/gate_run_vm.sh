#!/bin/bash
# gate_run_vm.sh (2026-09-29): the first run_vm.sh process was stopped while DT s0 trained the w=64 gate run, so that a
# failed w=64 gate continues at w=64 (GATE_MAX_W) instead of escalating to w=128. Wait for that training to finish,
# then exec the updated run_vm.sh (resume-safe: markers skip done steps, claims skip runs already going on this VM).
set -u
cd /root/job; OUT=/root/out; W=$(cat $OUT/W)
until [[ -f $OUT/markers/train_DT_s0_w$W.done ]]; do
  pgrep -f -- "--arm DT --seed 0 --w $W " >/dev/null || { sleep 20; [[ -f $OUT/markers/train_DT_s0_w$W.done ]] || { echo "gate run DT s0 w$W neither running nor done"; exit 1; }; }
  sleep 30
done
exec bash run_vm.sh
