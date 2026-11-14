#!/bin/bash
# Runs ON the L4 VM as root, cwd /root/soup = staged copy (harness/ + e5/). Usage: remote.sh JOBFILE
# JOBFILE: one job per line, lines prefixed "slotN|"; each slot runs its lines sequentially; 2 slots in parallel.
set -x
cd /root/soup
OUT=/root/runs/E5; mkdir -p $OUT
t0=$(date +%s)
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > $OUT/apt-$(hostname).log 2>&1
python3 -m pip --version >/dev/null 2>&1 || { curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q; }
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > $OUT/pip-$(hostname).log 2>&1 || { tail -20 $OUT/pip-$(hostname).log; exit 1; }
echo "pip_seconds $(( $(date +%s) - t0 ))"
export PYTHONPATH=/root/pylib:/root/soup/harness HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIR_HARNESS=/root/soup/harness
python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')" > $OUT/dl-$(hostname).log 2>&1
export HF_HUB_OFFLINE=1
JOBS=$1
for slot in $(cut -d'|' -f1 $JOBS | sort -u); do
  ( grep "^$slot|" $JOBS | cut -d'|' -f2- | while read -r line; do
      echo "[$(date -u +%H:%M:%S)] START $line" >> $OUT/jobs-$slot-$(hostname).log
      ( cd /root/soup/harness && bash -c "$line" ) >> $OUT/jobs-$slot-$(hostname).log 2>&1
      echo "[$(date -u +%H:%M:%S)] END rc=$? $line" >> $OUT/jobs-$slot-$(hostname).log
    done ) &
done
wait
echo "all done $(( $(date +%s) - t0 ))s"
