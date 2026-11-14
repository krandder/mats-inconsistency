#!/bin/bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 4:00:00
#SBATCH -J kgrpo-base-eval
#SBATCH -o /cluster/home/user/grpo-pilot/logs/%x-%j.out
# Base model (no adapter) on the held-out rows: the reference for every contrast.
source /cluster/home/user/grpo-pilot/code/ops/mats/grpo-pilot/common.sh
$PY $CODE/training/grpo_pilot_eval.py --model "$MODEL" --tag base --rows $DATA/test.jsonl $DATA/singles-external.jsonl \
  $DATA/validation.jsonl --out $BASE/eval
# LoRA path check on the smoke adapter (50 rows), if given
if [ -n "${SMOKE_ADAPTER:-}" ]; then
  $PY $CODE/training/grpo_pilot_eval.py --model "$MODEL" --adapter "$SMOKE_ADAPTER" --tag smoke-lora \
    --rows $DATA/singles-external.jsonl --limit 50 --out $BASE/eval-smoke
fi
