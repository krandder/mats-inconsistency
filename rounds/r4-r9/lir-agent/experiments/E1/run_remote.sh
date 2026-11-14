#!/bin/bash
# Runs ON the L4 VM as root, cwd /root/soup (= harness copy + e1/). Usage: run_remote.sh <jobfile in e1/jobs/>
# Job line: "<slot> <timeout_min> <name> <flags...>"; slot 0 jobs run first, all in parallel (warm starts);
# then slots 1 and 2 run their jobs sequentially, the two slots in parallel. No job starts after 72 min.
SOUP=${E1_SOUP:-/root/soup}
cd $SOUP
JOBS=$SOUP/e1/jobs/$1
OUT=${E1_OUT:-/root/runs/E1}
mkdir -p $OUT
T0=$(date +%s)
nvidia-smi > $OUT/nvidia-smi.txt 2>&1
if [ -z "${E1_LOCAL:-}" ]; then
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > $OUT/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > $OUT/pip.log 2>&1 || { tail -20 $OUT/pip.log; exit 1; }
fi
echo "setup_s $(( $(date +%s) - T0 ))" >> $OUT/jobs.log
[ -n "${E1_LOCAL:-}" ] || export PYTHONPATH=/root/pylib:$SOUP:$SOUP/e1 HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "${E1_LOCAL:-}" ] || python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')" > $OUT/hf.log 2>&1
C=${E1_C:-"--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset syco --groups 4 --group-size 8 --micro-batch 8 --threads 4 --eval-presets= --faith-swap-steps 0 --ckpt-dir /root/runs/ckpt"}
run_job() {
  local tmo=$1 name=$2; shift 2
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $((72*60)) ]; then echo "SKIP $name elapsed_s=$el" >> $OUT/jobs.log; return; fi
  local t1=$(date +%s)
  timeout --kill-after=30 ${tmo}m python3 e1/e1run.py $C "$@" --out $OUT/$name > $OUT/$name.log 2>&1 < /dev/null
  local rc=$?
  echo "$name rc=$rc wall_s=$(( $(date +%s) - t1 )) start_s=$(( t1 - T0 ))" >> $OUT/jobs.log
  python3 e1/e1_metrics.py $OUT/$name >> $OUT/jobs.log 2>&1 < /dev/null
}
for s in 0; do
  while read -r slot tmo name flags; do
    [ "$slot" = 0 ] && run_job $tmo $name $flags &
  done < $JOBS
done
wait
for s in 1 2; do
  ( grep "^$s " $JOBS | while read -r slot tmo name flags; do run_job $tmo $name $flags; done ) &
done
wait
echo "done total_s=$(( $(date +%s) - T0 ))" >> $OUT/jobs.log
