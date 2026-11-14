#!/bin/bash
# Runs ON the L4 VM as root. Usage: remote.sh KIND PRESET SEED   (KIND = F | R | X)
#   F: slot1 B0 (own SFT) ; slot2 M0 from B0's warm start           (fused layout)
#   R: slot1 P (own SFT) then twin from P's warm start ; slot2 M1 (own SFT, gating all)
#   X: slot1 B0 then M0 from B0's warm ; slot2 P then M1            (note_rel 0.4 setting; not used after 08:55Z)
#   R2: slot1 P (own SFT) ; slot2 twin from P's warm start           (added 08:55Z: 2 runs/GPU measured ~2x slower each)
#   MM: M1 for two seeds at once, SEED = two digits, e.g. 02
#   Y: B0 and P concurrently, each with its own SFT                   (note_rel 0.4 setting)
set -u
KIND=$1; PRESET=$2; SEED=$3
if [ "$KIND" = MM ]; then   # two seeds of M1 in parallel (second starts after the first's setup)
  S1=${SEED:0:1}; S2=${SEED:1:1}
  bash "$0" M1only $PRESET $S1 & sleep 300; bash "$0" M1only $PRESET $S2 & wait; exit 0
fi
cd /root/soup/harness
OUT=/root/runs/E4/$PRESET; mkdir -p $OUT
nvidia-smi > $OUT/nvidia-smi-$KIND-s$SEED.txt 2>&1
t0=$(date +%s)
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
[ -d /root/pylib/torch ] || python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup/harness HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B')" > /root/hf.log 2>&1
export HF_HUB_OFFLINE=1
echo "setup_s $(( $(date +%s) - t0 ))" > $OUT/setup-$KIND-s$SEED.txt
C=(--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset $PRESET --p-reward approval --sft-steps 100 --sft-batch 16
   --rl-steps 80 --groups 4 --group-size 8 --eval-every 20 --eval-n 96 --eval-presets "" --micro-batch 8 --threads 4
   --ckpt-dir /root/runs/ckpt/$PRESET --seed $SEED --e4-fixed-n 500)
X=/root/soup/E4/e4_train.py
TO=100m   # every run stops by itself; partial eval points are kept (e4_fixed.jsonl is appended per eval)
run() { local name=$1; shift; timeout $TO python3 $X "${C[@]}" "$@" --out $OUT/$name-s$SEED > $OUT/$name-s$SEED.log 2>&1;
        echo "$name rc=$? t=$(( $(date +%s) - t0 ))" >> $OUT/setup-$KIND-s$SEED.txt; }
waitwarm() { for i in $(seq 1 240); do [ -f /root/runs/ckpt/$PRESET/$1-s$SEED/warm.pt ] && sleep 5 && return 0; sleep 15; done; return 1; }
case $KIND in
  F) run B0 --arm B0 & (waitwarm B0 && run M0 --arm M0 --sft-steps 0 --init-from /root/runs/ckpt/$PRESET/B0-s$SEED/warm.pt) & wait ;;
  R) (run P --arm P; run twin --arm P --lr-action 0 --sft-steps 0 --init-from /root/runs/ckpt/$PRESET/P-s$SEED/warm.pt) &
     run M1 --arm P --e4-gating all & wait ;;
  R2) run P --arm P & (waitwarm P && run twin --arm P --lr-action 0 --sft-steps 0 --init-from /root/runs/ckpt/$PRESET/P-s$SEED/warm.pt) & wait ;;
  Y) run B0 --arm B0 & run P --arm P & wait ;;
  M1only) run M1 --arm P --e4-gating all ;;
  X) (run B0 --arm B0; ) & (waitwarm B0 && run M0 --arm M0 --sft-steps 0 --init-from /root/runs/ckpt/$PRESET/B0-s$SEED/warm.pt) &
     (run P --arm P; run M1 --arm P --e4-gating all) & wait ;;
esac
echo "done t=$(( $(date +%s) - t0 ))" >> $OUT/setup-$KIND-s$SEED.txt
