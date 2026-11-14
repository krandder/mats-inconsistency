#!/bin/bash
# Eight arms, identical flags except arm/labels/lambda.  Lanes run arms sequentially; two lanes per GPU.
# After each arm: rescore every kept epoch adapter on test on the same GPU.
# Usage: remote_queue.sh "<extra flags>" LANE0 LANE1 ...   where LANE = "gpu:ARM,ARM"
cd /workspace/consistency/code; export PYTHONPATH=.:scripts
D=/workspace/consistency/data/jev-disjoint-v1-filtered-20260919; M=../models/Qwen3-0.6B; R=/workspace/consistency/runs
EXTRA=$1; shift
COMMON="--data-root $D --model $M --device cuda:0 --max-length 1280 --batch-size 8 --eval-batch-size 8 --seed 20260919 --learning-rate 1e-5 --lora-rank 16 --lora-alpha 32 --epochs 6 --patience 2 $EXTRA"
declare -A FLAGS=(
 [A1_one_accuracy]="--arm accuracy --train-labels one"
 [A2_one_consistency_l0.3]="--arm consistency --train-labels one --lambda-consistency 0.3"
 [A3_one_consistency_l1]="--arm consistency --train-labels one --lambda-consistency 1.0"
 [A4_one_consistency_l3]="--arm consistency --train-labels one --lambda-consistency 3.0"
 [A5_one_prior_l1]="--arm prior --train-labels one --lambda-consistency 1.0"
 [A6_one_permuted_l1]="--arm permuted --train-labels one --lambda-consistency 1.0"
 [A7_full_accuracy]="--arm accuracy --train-labels full"
 [A8_full_consistency_l1]="--arm consistency --train-labels full --lambda-consistency 1.0"
)
mkdir -p $R; echo "QUEUE_START $(date -u +%FT%TZ) extra=[$EXTRA] lanes=[$*]" >> $R/status.txt
lane() {
  gpu=${1%%:*}; arms=${1#*:}
  for name in ${arms//,/ }; do
    [ -e $R/SKIP_$name ] && { echo "$name SKIPPED" >> $R/status.txt; continue; }
    echo "$name START gpu=$gpu $(date -u +%FT%TZ)" >> $R/status.txt
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_supervised_pairs.py $COMMON ${FLAGS[$name]} --output $R/$name > $R/$name.log 2>&1
    echo "$name EXIT=$? $(date -u +%FT%TZ)" >> $R/status.txt
    CUDA_VISIBLE_DEVICES=$gpu python scripts/rescore_checkpoints.py --run $R/$name --data-root $D --model $M --split test --device cuda:0 --max-length 1280 --batch-size 8 >> $R/$name.rescore.log 2>&1
    echo "$name RESCORE_EXIT=$? $(date -u +%FT%TZ)" >> $R/status.txt
  done
}
for l in "$@"; do lane "$l" & done
wait
echo "QUEUE_DONE $(date -u +%FT%TZ)" >> $R/status.txt
