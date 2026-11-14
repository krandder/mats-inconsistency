#!/bin/bash
# Submit the three arms, each as a chain of LINKS jobs (afterany), within the 6-GPU user cap (2 GPUs per arm).
# Usage: submit-arms.sh MAX_STEPS [LINKS]
set -euo pipefail
MAX_STEPS=$1; LINKS=${2:-3}
D=/cluster/home/user/grpo-pilot/code/ops/mats/grpo-pilot
for ARM in acc acc_cons cons; do
  prev=""
  for i in $(seq 1 "$LINKS"); do
    dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable $dep -J kgrpo-$ARM --export=ALL,ARM=$ARM,RUN=arm-$ARM,MAX_STEPS=$MAX_STEPS $D/arm.sbatch)
    echo "$ARM link $i job $prev"
  done
done
