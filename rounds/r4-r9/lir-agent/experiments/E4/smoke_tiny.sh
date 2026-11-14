#!/bin/bash
# CPU pipeline proof on the random tiny model: every E4 arm config end to end.
cd /home/agent/workspaces/eurisko-study/lir-agent/harness && source scripts/env.sh
O=/tmp/lir-agent/E4/smoke; mkdir -p $O; rm -rf $O/*/ 2>/dev/null
T=(--model tiny --preset syco16 --p-reward approval --sft-steps 2 --sft-batch 4 --rl-steps 2 --groups 2 --group-size 2
   --eval-every 1 --eval-n 4 --eval-presets "" --threads 2 --ckpt-dir /tmp/lir-agent/E4/ckpt --e4-fixed-n 20)
X=../experiments/E4/e4_train.py
for a in B0 M0 P; do timeout 900 python3 $X "${T[@]}" --arm $a --out $O/$a > $O/$a.log 2>&1; echo "$a rc=$?"; done
timeout 900 python3 $X "${T[@]}" --arm P --e4-gating all --out $O/M1 > $O/M1.log 2>&1; echo "M1 rc=$?"
timeout 900 python3 $X "${T[@]}" --arm P --lr-action 0 --sft-steps 0 --init-from /tmp/lir-agent/E4/ckpt/P/warm.pt --out $O/twin > $O/twin.log 2>&1; echo "twin rc=$?"
