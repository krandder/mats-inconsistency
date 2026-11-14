#!/bin/bash
# Runs ON the L4 VM as root; /root/soup = frozen copy of harness/ plus e7/ (this directory).
# Usage: remote.sh MODE   MODE in w1a w1b w2a w2b w2c w3
set -u
MODE=$1
cd /root/soup
OUT=/root/runs/E7; mkdir -p $OUT /root/runs/ckpt /root/runs/an
LOG=$OUT/remote-$MODE.log
# (no tee: a process substitution makes a bare `wait` hang; stdout goes to the driver log)
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
echo "setup_seconds $(( $(date +%s) - t0 ))"
export PYTHONPATH=/root/pylib:/root/soup:/root/soup/e7 HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B')" > /dev/null 2>&1
C=(--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset easy --sft-batch 16 --groups 4 --group-size 8
   --eval-n 128 --eval-presets "" --micro-batch 8 --threads 4 --ckpt-dir /root/runs/ckpt)
argsof() {  # label -> harness arm + gating
  case $1 in P-all) echo "--arm P --gating all";; P-role) echo "--arm P --gating role";; B0) echo "--arm B0 --gating all";; esac
}
warm() {  # label seed
  local L=$1 S=$2
  python3 e7/run_arm.py "${C[@]}" $(argsof $L) --label $L --seed $S --sft-steps 80 --rl-steps 0 \
    --out $OUT/warm-$L-s$S > $OUT/warm-$L-s$S.log 2>&1
  echo "warm $L s$S rc=$? t=$(( $(date +%s) - t0 ))"
}
rl() {  # label seed
  local L=$1 S=$2
  python3 e7/run_arm.py "${C[@]}" $(argsof $L) --label $L --seed $S --sft-steps 0 \
    --init-from /root/runs/ckpt/warm-$L-s$S/pre_rl.pt --rl-steps 60 --eval-every 30 \
    --out $OUT/$L-s$S > $OUT/$L-s$S.log 2>&1
  echo "rl $L s$S rc=$? t=$(( $(date +%s) - t0 ))"
  rm -f /root/runs/ckpt/$L-s$S/pre_rl.pt     # identical to the warm checkpoint
}
lane() {  # fn "L S" "L S" ...
  local fn=$1; shift
  for x in "$@"; do $fn $x; done
}
case $MODE in
  w1a)
    lane warm "P-all 0" "P-all 2" "P-all 4" "B0 0" &
    lane warm "P-all 1" "P-all 3" "B0 1" &
    wait
    mkdir -p /root/runs/g0ck
    for s in 0 1 2 3 4; do ln -sfn /root/runs/ckpt/warm-P-all-s$s /root/runs/g0ck/P-all-s$s; done
    for k in 0 1; do
      python3 e7/analyze.py --ckroot /root/runs/g0ck --out /root/runs/an --arms P-all --tags pre_rl \
        --fresh-arms P-all --fresh-tags pre_rl --part $k --nparts 2 > $OUT/gate0-part$k.log 2>&1 &
    done
    wait; echo "gate0 done t=$(( $(date +%s) - t0 ))" ;;
  w1b)
    lane warm "P-role 0" "P-role 2" "P-role 4" "B0 3" &
    lane warm "P-role 1" "P-role 3" "B0 2" "B0 4" &
    wait ;;
  w2a|w2b|w2c)
    case $MODE in
      w2a) A=P-all;  AN=(--arms P-all --fresh-arms P-all --fresh-tags post_rl) ;;
      w2b) A=P-role; AN=(--arms P-role B0 --fresh-arms B0 --fresh-tags pre_rl) ;;
      w2c) A=B0;     AN=(--arms B0 --fresh-arms B0 --fresh-tags post_rl) ;;
    esac
    lane rl "$A 0" "$A 2" "$A 4" &
    lane rl "$A 1" "$A 3" &
    wait
    for L in P-all P-role B0; do for s in 0 1 2 3 4; do
      d=/root/runs/ck3/$L-s$s; mkdir -p $d
      [ -f /root/runs/ckpt/warm-$L-s$s/pre_rl.pt ] && ln -sfn /root/runs/ckpt/warm-$L-s$s/pre_rl.pt $d/pre_rl.pt
      for t in rl30 post_rl; do [ -f /root/runs/ckpt/$L-s$s/$t.pt ] && ln -sfn /root/runs/ckpt/$L-s$s/$t.pt $d/$t.pt; done
    done; done
    for k in 0 1; do
      python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an "${AN[@]}" --part $k --nparts 2 > $OUT/an-$MODE-part$k.log 2>&1 &
    done
    wait; echo "analysis done t=$(( $(date +%s) - t0 ))" ;;
  w2m1|w2m2)
    if [ $MODE = w2m1 ]; then
      lane rl "P-all 0" "P-all 2" "B0 1" &
      lane rl "P-all 1" "B0 0" &
      SP=(0 1 2); SB=(0 1)
    else
      lane rl "B0 2" "B0 4" "P-all 4" &
      lane rl "B0 3" "P-all 3" &
      SP=(3 4); SB=(2 3 4)
    fi
    wait
    for L in P-all B0; do for s in 0 1 2 3 4; do
      d=/root/runs/ck3/$L-s$s; mkdir -p $d
      [ -f /root/runs/ckpt/warm-$L-s$s/pre_rl.pt ] && ln -sfn /root/runs/ckpt/warm-$L-s$s/pre_rl.pt $d/pre_rl.pt
      for t in rl30 post_rl; do [ -f /root/runs/ckpt/$L-s$s/$t.pt ] && ln -sfn /root/runs/ckpt/$L-s$s/$t.pt $d/$t.pt; done
    done; done
    python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an --arms P-all --seeds ${SP[@]} --fresh-arms P-all \
      --fresh-tags post_rl --pred-seeds 1 > $OUT/an-$MODE-P.log 2>&1 &
    python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an --arms B0 --seeds ${SB[@]} --fresh-arms B0 \
      --fresh-tags post_rl --pred-seeds 1 > $OUT/an-$MODE-B.log 2>&1 &
    wait; echo "analysis done t=$(( $(date +%s) - t0 ))" ;;
  w2c2)
    lane rl "B0 0" "B0 1" "B0 2" "B0 3"
    for L in P-all P-role B0; do for s in 0 1 2 3 4; do
      d=/root/runs/ck3/$L-s$s; mkdir -p $d
      [ -f /root/runs/ckpt/warm-$L-s$s/pre_rl.pt ] && ln -sfn /root/runs/ckpt/warm-$L-s$s/pre_rl.pt $d/pre_rl.pt
      for t in rl30 post_rl; do [ -f /root/runs/ckpt/$L-s$s/$t.pt ] && ln -sfn /root/runs/ckpt/$L-s$s/$t.pt $d/$t.pt; done
    done; done
    for k in 0 1; do
      python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an --arms B0 --fresh-arms B0 --fresh-tags post_rl \
        --part $k --nparts 2 > $OUT/an-$MODE-part$k.log 2>&1 &
    done
    wait; echo "analysis done t=$(( $(date +%s) - t0 ))" ;;
  w2r3)
    lane rl "P-role 0" "P-role 2" &
    lane rl "P-role 1" &
    wait
    for s in 0 1 2; do
      d=/root/runs/ck3/P-role-s$s; mkdir -p $d
      ln -sfn /root/runs/ckpt/warm-P-role-s$s/pre_rl.pt $d/pre_rl.pt
      for t in rl30 post_rl; do [ -f /root/runs/ckpt/P-role-s$s/$t.pt ] && ln -sfn /root/runs/ckpt/P-role-s$s/$t.pt $d/$t.pt; done
    done
    for k in 0 1; do
      python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an --arms P-role --seeds 0 1 2 --fresh-arms none \
        --part $k --nparts 2 > $OUT/an-$MODE-part$k.log 2>&1 &
    done
    wait; echo "analysis done t=$(( $(date +%s) - t0 ))" ;;
  w3)
    for k in 0 1; do
      python3 e7/analyze.py --ckroot /root/runs/ck3 --out /root/runs/an --part $k --nparts 2 > $OUT/an-w3-part$k.log 2>&1 &
    done
    wait; echo "analysis done t=$(( $(date +%s) - t0 ))" ;;
esac
echo "remote $MODE end t=$(( $(date +%s) - t0 ))"
