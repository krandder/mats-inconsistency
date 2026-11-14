#!/usr/bin/env bash
# Remote run: vLLM rollout server on GPU 3, GRPO trainer (DDP, 3 processes) on GPUs 0-2.
# Usage on the box: bash run.sh [extra trainer args, e.g. --resume]   (start.py launches it under nohup)
# GPU split rationale: TRL 1.13 "server" mode keeps the generation engine off the training GPUs; the trainer
# merges the LoRA into the base weights and pushes them to the server over NCCL before every generation batch
# (trl/generation/vllm_generation.py sync_weights), so the server always samples from the current policy.
# Colocate mode (vLLM inside each trainer process) would need a second 28 GB copy of the 14B per GPU and is not
# used. If a TRL upgrade ever drops server mode, switch to --vllm-mode colocate with --vllm-gpu-memory-utilization 0.3.
set -uo pipefail
ROOT=/workspace/turtel
MODEL=$ROOT/models/DeepSeek-R1-Distill-Qwen-14B
RUN=$ROOT/runs/turtel-14b-grpo
PORT=8000
mkdir -p $RUN
export HF_HOME=$ROOT/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
ARM_ARGS=$(python3 -c "import json; print(' '.join(json.load(open('$ROOT/manifest.json'))['arm']['trainer_args']))")
echo "$(date -u +%FT%TZ) RUN_SH start args: $ARM_ARGS $*" >> $RUN/status.txt

# 1. rollout server (weights are replaced by the trainer; the checkpoint on disk only seeds it)
if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=3 nohup trl vllm-serve --model $MODEL --tensor-parallel-size 1 --data-parallel-size 1 \
    --host 127.0.0.1 --port $PORT --gpu-memory-utilization 0.90 --max-model-len 12288 --dtype bfloat16 \
    > $RUN/vllm-server.log 2>&1 &
  echo $! > $RUN/vllm-server.pid
fi
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 10; done
if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) RUN_FAILED vllm server never became healthy" >> $RUN/status.txt; exit 1
fi
echo "$(date -u +%FT%TZ) VLLM_READY" >> $RUN/status.txt

# 2. trainer on GPUs 0-2 (DDP; LoRA keeps optimizer state small, no ZeRO needed)
cd $ROOT/code
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29511 training/turtel_grpo.py \
  --model $MODEL --train $ROOT/data/train.jsonl --validation $ROOT/data/validation.jsonl \
  --test-set $ROOT/data/test.parquet --run-dir $RUN --vllm-server-url http://127.0.0.1:$PORT --num-processes 3 \
  $ARM_ARGS "$@" > $RUN/trainer.log 2>&1
RC=$?
echo "$(date -u +%FT%TZ) TRAINER EXIT=$RC" >> $RUN/status.txt
exit $RC
