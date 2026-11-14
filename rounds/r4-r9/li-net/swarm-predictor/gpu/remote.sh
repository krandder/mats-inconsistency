#!/bin/bash
# Runs ON the L4 VM as root. /root/soup = frozen copy of: harness/ (env, model; unmodified) + swarm/ (this code).
# Usage: remote.sh JOBFILE   JOBFILE (relative to /root/soup): one run_swarm.py argument list per line. Lines between
#        "---" separators form a group: the lines of a group run in parallel on the one GPU, groups run one after
#        another. Arguments must not contain spaces. Lines starting with # are comments.
set -u
cd /root/soup
JOBS=$1
mkdir -p /root/runs/swarm /root/text /root/ckpt
H=$(hostname)
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 \
  || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup/harness:/root/soup/swarm LIR_HARNESS=/root/soup/harness HF_HOME=/root/hf \
  HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in $(grep -oE 'Qwen/Qwen3-[0-9.]+B' "$JOBS" | sort -u) Qwen/Qwen3-0.6B; do
  python3 -c "from huggingface_hub import snapshot_download as s; s('$m')" > /dev/null 2>&1 || echo "download $m failed"
done
echo "setup_seconds $(( $(date +%s) - t0 ))" | tee /root/runs/swarm/setup-$H.txt
python3 -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)" \
  | tee -a /root/runs/swarm/setup-$H.txt
k=0
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  if [[ $line == ---* ]]; then wait; echo "group done t=$(( $(date +%s) - t0 ))" >> /root/runs/swarm/jobs-$H.txt; continue; fi
  k=$((k+1))
  read -ra A <<< "$line"
  ( python3 swarm/run_swarm.py "${A[@]}" > /root/runs/swarm/job$k-$H.log 2>&1
    echo "job$k rc=$? t=$(( $(date +%s) - t0 )) args: $line" >> /root/runs/swarm/jobs-$H.txt ) &
  sleep 15
done < "$JOBS"
wait
echo "all jobs done t=$(( $(date +%s) - t0 ))" | tee -a /root/runs/swarm/jobs-$H.txt
