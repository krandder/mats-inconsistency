#!/bin/bash
# E6 at toy scale (PREREG deviation 09:15Z): X then yoked b_X for one arm and seed. Usage: run_tiny.sh c|d SEED
set -u
X=$1; S=$2
cd "$(dirname "$0")/../../harness"
source scripts/env.sh
E=../experiments/E6; CK=/tmp/lir-agent/e6/ckpt-tiny
ARM=$([ "$X" = c ] && echo P || echo B0)
C="--model tiny --lr 1e-3 --threads 2 --micro-batch 8 --ckpt-dir $CK --eval-every 25 --rl-steps 200"
timeout 3h python3 $E/e6run.py --mode train --arm $ARM --seed $S $C --sft-steps 40 --sft-batch 16 --groups 4 --group-size 8 \
  --out $E/runs/$X-s$S > /tmp/lir-agent/e6/tiny-$X-s$S.log 2>&1 || { echo "FAIL train $X $S" >> $E/runs/FAILS; exit 1; }
timeout 3h python3 $E/e6run.py --mode replay --arm $ARM --seed $S $C --stream $E/runs/$X-s$S/stream \
  --warm $CK/$X-s$S/warm.pt --out $E/runs/b$X-s$S > /tmp/lir-agent/e6/tiny-b$X-s$S.log 2>&1 || echo "FAIL replay $X $S" >> $E/runs/FAILS
