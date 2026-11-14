#!/bin/bash
#SBATCH -p compute
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 23:50:00
#SBATCH -o /cluster/home/user/grpo-pilot/logs/%x-%j.out
# One arm of the GRPO consistency pilot, or the smoke run. Env: ARM (acc|cons|acc_cons), RUN (run dir name),
# MAX_STEPS, EXTRA (extra trainer flags, smoke only). Re-submitted in a dependency chain; each link resumes from the
# latest checkpoint and exits at once when the run has STOP or RUN_DONE and its eval is done.
source /cluster/home/user/grpo-pilot/code/ops/mats/grpo-pilot/common.sh
RUN_DIR=$BASE/runs/$RUN
mkdir -p "$RUN_DIR"
if [ -f "$RUN_DIR/EVAL_DONE" ]; then echo "already evaluated"; exit 0; fi
PORT=$((20000 + SLURM_JOB_ID % 20000)); GROUP_PORT=$((40000 + SLURM_JOB_ID % 20000))
if [ ! -f "$RUN_DIR/RUN_DONE" ] && [ ! -f "$RUN_DIR/STOP" ]; then
  start_server "${GPUS[0]}" $PORT "$RUN_DIR/vllm-server-$SLURM_JOB_ID.log"
  set +e
  CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY $CODE/training/grpo_consistency.py --arm "$ARM" --model "$MODEL" \
    --train $DATA/train.jsonl --validation $DATA/validation.jsonl --run-dir "$RUN_DIR" --resume \
    --max-steps "$MAX_STEPS" --vllm-server-url http://127.0.0.1:$PORT --vllm-group-port $GROUP_PORT ${EXTRA:-}
  rc=$?
  set -e
  kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
  echo "trainer exit $rc"
  if [ $rc -ne 0 ]; then exit $rc; fi
fi
if [ -d "$RUN_DIR/final" ] && { [ -f "$RUN_DIR/RUN_DONE" ] || [ -f "$RUN_DIR/STOP" ]; }; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $CODE/training/grpo_pilot_eval.py --model "$MODEL" --adapter "$RUN_DIR/final" \
    --tag "$RUN" --rows $DATA/test.jsonl $DATA/singles-external.jsonl --out $BASE/eval ${EVAL_EXTRA:-}
  date -u +%FT%TZ > "$RUN_DIR/EVAL_DONE"
fi
