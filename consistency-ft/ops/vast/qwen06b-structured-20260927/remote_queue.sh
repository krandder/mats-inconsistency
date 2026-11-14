#!/bin/bash
# Arms for the structured-pairs run, identical flags except arm/labels/lambda.  Lanes run arms in sequence
# on the one GPU.  The zero-shot base model is scored on test inside A1 (--zero-shot-test).  Each arm keeps
# the best-validation epoch (mean 12-form log loss) and scores test once with it.
# Usage: remote_queue.sh "<extra flags>" "ARM,ARM" "ARM,..."
cd /workspace/consistency/code; export PYTHONPATH=.:scripts
D=/workspace/consistency/data/qwen06b-structured; M=../models/Qwen3-0.6B; R=/workspace/consistency/runs
EXTRA=$1; shift
COMMON="--data-root $D --model $M --device cuda:0 --max-length 4096 --shared-prefix --select-on full --batch-size 8 --eval-batch-size 8 --seed 20260927 --learning-rate 3e-5 --lora-rank 16 --lora-alpha 32 --epochs 8 --patience 2 $EXTRA"
declare -A FLAGS=(
 [A1_full_accuracy]="--arm accuracy --train-labels full --zero-shot-test"
 [A2_full_consistency_l1]="--arm consistency --train-labels full --lambda-consistency 1.0"
 [A3_full_consistency_l3]="--arm consistency --train-labels full --lambda-consistency 3.0"
 [A4_full_prior_l3]="--arm prior --train-labels full --lambda-consistency 3.0"
 [A5_one_consistency_l3]="--arm consistency --train-labels one --lambda-consistency 3.0"
)
mkdir -p $R; echo "QUEUE_START $(date -u +%FT%TZ) extra=[$EXTRA] lanes=[$*]" >> $R/status.txt
lane() {
  for name in ${1//,/ }; do
    [ -e $R/SKIP_$name ] && { echo "$name SKIPPED" >> $R/status.txt; continue; }
    echo "$name START $(date -u +%FT%TZ)" >> $R/status.txt
    python scripts/train_supervised_pairs.py $COMMON ${FLAGS[$name]} --output $R/$name > $R/$name.log 2>&1
    echo "$name EXIT=$? $(date -u +%FT%TZ)" >> $R/status.txt
  done
}
for l in "$@"; do lane "$l" & done
wait
echo "QUEUE_DONE $(date -u +%FT%TZ)" >> $R/status.txt
