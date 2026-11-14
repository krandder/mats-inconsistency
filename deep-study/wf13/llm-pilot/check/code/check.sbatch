#!/bin/bash
#SBATCH --job-name=w13-llm-check
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:40:00
#SBATCH --output=/cluster/home/user/wave13/llm-pilot-check/job-%j.log
D=/cluster/home/user/wave13/llm-pilot-check
W=/cluster/home/user/wave13/llm-pilot
PY=/cluster/home/user/grpo-pilot/venv/bin/python
M=$W/hf/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
cd $D || exit 1
mkdir -p lora out logs
date -u; nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -B make_pure.py out/lora_pure.jsonl
[ -d lora/pure ] || $PY -B $W/code2/train_lora.py $M out/lora_pure.jsonl lora/pure 2 5 > logs/lora_pure.log 2>&1
date -u
[ -d lora/bias1_s11 ] || $PY -B $W/code2/train_lora.py $M $W/out/main/lora_bias1.jsonl lora/bias1_s11 2 11 > logs/lora_bias1_s11.log 2>&1
date -u
$PY -B fresh.py out/fresh.jsonl ${NPROB:-60} > logs/fresh.log 2>&1
date -u
