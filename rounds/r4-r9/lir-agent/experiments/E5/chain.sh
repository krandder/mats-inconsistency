#!/bin/bash
# One E5a chain on the VM (cwd /root/soup/harness): warm start in goal_mode GM, then one RL run per reward
# from that warm start, then e5 metrics and (visible/shuffled) the format control. Usage: chain.sh ARM GM SEED REWARDS
A=$1; GM=$2; S=$3; REW=${4:-approval,random}
case $A in B0) AF="--arm B0"; FU=1 ;; D1) AF="--arm P --alpha 1.0"; FU=0 ;; P) AF="--arm P"; FU=0 ;; esac
O=/root/runs/E5/E5a; CK=/root/runs/ckpt; mkdir -p $O
C="--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset easy --goal mixed --sft-steps 60 --sft-batch 16 --groups 4 --group-size 8 --eval-every 32 --eval-n 128 --eval-presets= --micro-batch 4 --threads 4 --ckpt-dir $CK"
W=$A-$GM-s$S-warm
python3 -m protocol.train $C $AF --p-goal-mode $GM --seed $S --rl-steps 0 --out $O/$W > $O/$W.log 2>&1 || echo "WARM FAIL $W"
FMT="warm=$CK/$W/warm.pt"
for R in ${REW//,/ }; do
  N=$A-$GM-$R-s$S
  python3 -m protocol.train $C $AF --p-goal-mode $GM --seed $S --rl-steps 64 --sft-steps 0 --init-from $CK/$W/warm.pt --p-reward $R --out $O/$N > $O/$N.log 2>&1 || echo "RL FAIL $N"
  python3 /root/soup/e5/e5a_metrics.py $O/$N > $O/$N.metrics.log 2>&1
  FMT="$FMT $R=$CK/$N/final.pt"
  rm -f $CK/$N/warm.pt
done
if [ "$GM" != deleted ]; then
  python3 /root/soup/e5/e5a_format.py --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 --fused $FU --ckpts $FMT --out $O/$A-$GM-s$S.format.json > $O/$A-$GM-s$S.format.log 2>&1
fi
