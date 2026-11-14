#!/bin/bash
# E6: X (train, stream logged) then b_X (yoked replay) for one arm and seed. Usage: pipeline.sh c|d SEED
set -u
X=$1; S=$2
cd "$(dirname "$0")/../../harness"
source scripts/env.sh
E=../experiments/E6
ARM=$([ "$X" = c ] && echo P || echo B0)
CK=/tmp/lir-agent/e6/ckpt
F="--sft-steps 6 --sft-batch 8 --rl-steps 24 --groups 2 --group-size 4 --eval-every 6 --threads 2 --micro-batch ${MB:-8} --ckpt-dir $CK"
timeout 4h python3 $E/e6run.py --mode train --arm $ARM --seed $S $F --out $E/runs/$X-s$S > /tmp/lir-agent/e6/$X-s$S.log 2>&1 || echo "FAIL train $X $S" >> $E/runs/FAILS
timeout 4h python3 $E/e6run.py --mode replay --arm $ARM --seed $S --rl-steps 24 --eval-every 6 --threads 2 --micro-batch ${MB:-8} --ckpt-dir $CK \
  --stream $E/runs/$X-s$S/stream --warm $CK/$X-s$S/warm.pt --out $E/runs/b$X-s$S > /tmp/lir-agent/e6/b$X-s$S.log 2>&1 || echo "FAIL replay $X $S" >> $E/runs/FAILS
