#!/bin/bash
# Runs ON the L4 VM as root; cwd /root/soup = staged copy (harness/ + E3/). Args: KIND(roles|fused) SEED [RL_STEPS] [SFT_STEPS] [N_STATES]
# Hard internal deadline: everything stops by DEADLINE_MIN after start; evals write per-pair JSON as they go.
set -x
KIND=$1; S=$2; RL=${3:-80}; SFT=${4:-80}; NST=${5:-240}; DEADLINE_MIN=${DEADLINE_MIN:-108}
T0=$(date +%s); left() { echo $(( DEADLINE_MIN*60 - ($(date +%s)-T0) )); }
OUT=/root/runs/E3/s$S; CK=/root/ck; mkdir -p $OUT $CK
nvidia-smi > $OUT/nvidia-smi-$KIND.txt 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup/harness LIR_HARNESS=/root/soup/harness HF_HOME=/root/hf HF_HUB_OFFLINE=0 \
  TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')" > /root/dl.log 2>&1
export HF_HUB_OFFLINE=1
E=/root/soup/E3
echo "setup_s $(( $(date +%s)-T0 ))" >> $OUT/times-$KIND.txt
TR="python3 $E/e3_train.py --seed $S --rl-steps $RL --micro-batch 4 --threads 2"
PAIRS_FULL=c000:c000,c100:c100,c100:c000,c050:c050,c050:c000,c025:c025,c025:c000
EV="python3 $E/e3_eval.py --n-states $NST --n-cands 64 --n-probe-states 200 --n-probe-plans 16"
if [[ $KIND == roles || $KIND == rolestrain ]]; then
  timeout 30m $TR --arm P --warm-only --sft-steps $SFT --out $OUT/warm --ckpt $CK/warm > $OUT/warm.log 2>&1
  echo "warm rc=$? t=$(( $(date +%s)-T0 ))" >> $OUT/times-$KIND.txt
  W=$CK/warm/c000.pt
  TL=$(( $(left) - 30*60 )); RB=$(( $(date +%s) + TL - 150 ))
  timeout $TL $TR --arm P  --init-from $W --rl-deadline $RB --reward-mode env      --out $OUT/P    --ckpt $CK/P    > $OUT/P.log 2>&1 &
  timeout $TL $TR --arm P  --init-from $W --rl-deadline $RB --reward-mode forecast --out $OUT/RAPc --ckpt $CK/RAPc > $OUT/RAPc.log 2>&1 &
  timeout $TL $TR --arm FB --init-from $W --rl-deadline $RB --reward-mode forecast --out $OUT/RAPf --ckpt $CK/RAPf > $OUT/RAPf.log 2>&1 &
  wait
  echo "rl done t=$(( $(date +%s)-T0 ))" >> $OUT/times-$KIND.txt
  [[ $KIND == rolestrain ]] && exit 0
  TL=$(( $(left) - 60 ))
  timeout $TL $EV --ckpt $CK/P    --out $OUT/P/eval    --pairs $PAIRS_FULL > $OUT/P/eval.log 2>&1 &
  timeout $TL $EV --ckpt $CK/RAPc --out $OUT/RAPc/eval --pairs c100:c100,c100:c000,c050:c050,c050:c000,c025:c025,c025:c000 > $OUT/RAPc/eval.log 2>&1 &
  timeout $TL $EV --ckpt $CK/RAPf --out $OUT/RAPf/eval --pairs c100:c100,c050:c050,c025:c025 > $OUT/RAPf/eval.log 2>&1 &
  wait
else
  TL=$(( $(left) - 30*60 )); RB=$(( $(date +%s) + TL - 150 ))
  timeout $TL $TR --arm M0 --rl-deadline $RB --p-goal-mode visible --sft-steps $SFT --out $OUT/M0g --ckpt $CK/M0g > $OUT/M0g.log 2>&1 &
  timeout $TL $TR --arm B0 --rl-deadline $RB --preset hardsyco --p-reward approval --sft-steps $SFT --out $OUT/B0s --ckpt $CK/B0s > $OUT/B0s.log 2>&1 &
  wait
  echo "rl done t=$(( $(date +%s)-T0 ))" >> $OUT/times-$KIND.txt
  TL=$(( $(left) - 60 ))
  timeout $TL $EV --fused 1 --goal-mode visible --ckpt $CK/M0g --out $OUT/M0g/eval --pairs $PAIRS_FULL > $OUT/M0g/eval.log 2>&1 &
  timeout $TL $EV --fused 1 --preset hardsyco   --ckpt $CK/B0s --out $OUT/B0s/eval --pairs $PAIRS_FULL > $OUT/B0s/eval.log 2>&1 &
  wait
fi
echo "all done t=$(( $(date +%s)-T0 ))" >> $OUT/times-$KIND.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> $OUT/nvidia-smi-$KIND.txt 2>&1
exit 0
