# Sourced by the GRPO pilot job scripts (MATS cluster, Slurm). Paths under /cluster/home/user/grpo-pilot.
set -euo pipefail
BASE=/cluster/home/user/grpo-pilot
CODE=$BASE/code
DATA=$BASE/data
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING OMP_NUM_THREADS=8
MODEL_REV=1cfa9a7208912126459214e8b04321603b3df60c
MODEL=$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots/$MODEL_REV
PY=$BASE/venv/bin/python
TRL=$BASE/venv/bin/trl
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

start_server() {  # $1 = gpu id, $2 = port, $3 = log
  CUDA_VISIBLE_DEVICES=$1 $TRL vllm-serve --model "$MODEL" --port "$2" --gpu_memory_utilization 0.85 \
    --max_model_len 4096 --enable_prefix_caching true > "$3" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$2/health/" > /dev/null; then echo "vllm server up after $((i*5))s"; return 0; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "vllm server died"; tail -50 "$3"; return 1; fi
    sleep 5
  done
  echo "vllm server not healthy after 15 min"; return 1
}
