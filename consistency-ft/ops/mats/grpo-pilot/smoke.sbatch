#!/bin/bash
#SBATCH -p compute
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 6:00:00
#SBATCH -J kgrpo-smoke
#SBATCH -o /cluster/home/user/grpo-pilot/logs/%x-%j.out
# Smoke: A+C arm, 12 steps, then resume to 20 (checkpoint/resume check), validation on 20 pairs every 10 steps;
# then the full base eval on GPU 0 and the LoRA eval path on 50 rows on GPU 1.
source /cluster/home/user/grpo-pilot/code/ops/mats/grpo-pilot/common.sh
RUN_DIR=$BASE/runs/smoke-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
PORT=$((20000 + SLURM_JOB_ID % 20000)); GROUP_PORT=$((40000 + SLURM_JOB_ID % 20000))
python_ver=$($PY -c "import torch,trl,transformers,peft;print(torch.__version__,trl.__version__,transformers.__version__,peft.__version__, torch.cuda.is_available())") || true
echo "stack: $python_ver"
start_server "${GPUS[0]}" $PORT "$RUN_DIR/vllm-server.log"
for STEPS in 12 20; do
  rm -f "$RUN_DIR/RUN_DONE"
  CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY $CODE/training/grpo_consistency.py --arm acc_cons --model "$MODEL" \
    --train $DATA/train.jsonl --validation $DATA/validation.jsonl --run-dir "$RUN_DIR" --resume \
    --max-steps $STEPS --vllm-server-url http://127.0.0.1:$PORT --vllm-group-port $GROUP_PORT \
    --eval-every 10 --eval-limit 20 || { echo "TRAINER FAILED at stage $STEPS"; kill $SERVER_PID; exit 1; }
done
kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $CODE/training/grpo_pilot_eval.py --model "$MODEL" --tag base \
  --rows $DATA/test.jsonl $DATA/singles-external.jsonl $DATA/validation.jsonl --out $BASE/eval &
E1=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY $CODE/training/grpo_pilot_eval.py --model "$MODEL" --adapter "$RUN_DIR/final" \
  --tag smoke-lora --rows $DATA/singles-external.jsonl --limit 50 --out $RUN_DIR/eval &
E2=$!
wait $E1; echo "base eval exit $?"; wait $E2; echo "lora eval exit $?"
echo SMOKE_DONE
