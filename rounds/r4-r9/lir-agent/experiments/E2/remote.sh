#!/bin/bash
# Runs ON the L4 VM as root in /root/soup (= harness copy + e2/). Arg: job file (relative to /root/soup/e2/jobs).
# Job line: LANE NAME LEAK ARGS...   (two lanes a, b run in parallel; jobs within a lane run in order)
set -u
cd /root/soup
JOBS=/root/soup/e2/jobs/$1
OUT=/root/runs/E2
mkdir -p $OUT
t0=$(date +%s)
nvidia-smi > $OUT/nvidia-smi-$1.json.txt 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup:/root/soup/e2 LIR_HARNESS=/root/soup HF_HOME=/root/hf HF_HUB_OFFLINE=0 \
  TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')" > /root/prefetch.log 2>&1
echo "{\"setup_s\": $(( $(date +%s) - t0 ))}" > $OUT/setup-$1.json
C="--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --micro-batch 8 --threads 4 --sft-steps 60 --sft-batch 16 --rl-steps 80 --groups 4 --group-size 8 --eval-every 40 --eval-n 96 --faith-swap-steps 0 --ckpt-dir /root/runs/ckpt"
lane() {
  grep "^$1 " $JOBS | while read -r _ name leak args; do
    d=$OUT/$name; mkdir -p $d
    if [[ $args == *"--init-from"* ]]; then
      w=$(sed 's/.*--init-from \([^ ]*\).*/\1/' <<<"$args"); n=0
      until [ -f "$w" ] || [ $n -gt 180 ]; do sleep 10; n=$((n+1)); done
      sleep 5
    fi
    s=$(date +%s)
    python3 e2/e2_train.py --leak $leak $C $args --out $d > $d/train.log 2>&1; rc=$?
    m=$(date +%s)
    python3 e2/posthoc.py $d /root/runs/ckpt/$name --leak $leak --device cuda > $d/posthoc.log 2>&1; rc2=$?
    e=$(date +%s)
    python3 - "$d" "$name" "$rc" "$rc2" "$((m-s))" "$((e-m))" <<'PY'
import json, sys
d, name, rc, rc2, ts, tp = sys.argv[1:]
tail = open(d + "/train.log", errors="replace").read().splitlines()[-25:]
ptail = open(d + "/posthoc.log", errors="replace").read().splitlines()[-10:]
json.dump({"name": name, "train_rc": int(rc), "posthoc_rc": int(rc2), "train_s": int(ts), "posthoc_s": int(tp),
           "train_log_tail": tail, "posthoc_log_tail": ptail}, open(d + "/status.json", "w"), indent=1)
PY
  done
}
lane a & lane b & wait
echo "{\"total_s\": $(( $(date +%s) - t0 ))}" > $OUT/done-$1.json
