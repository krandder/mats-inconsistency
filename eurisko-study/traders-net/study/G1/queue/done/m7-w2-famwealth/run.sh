#!/bin/bash
# M7 w2 stage 1: DeepSeek bias + per-family wealth prior (+ random-group control, same-code none/DeepSeek). 42 runs, 7 parallel.
set -u
export M7_SUBSTRATE=$PWD/substrate
mkdir -p $OUT/runs $OUT/logs
nvidia-smi -L > $OUT/gpu.txt 2>&1
sha256sum moe_w2.py substrate/*.py > $OUT/SHA256SUMS.txt
one() {
  arm=$1; E=$2; s=$3
  python moe_w2.py --arm $arm --E $E --seed $s --T 24000 --B 8 --device cuda --out $OUT/runs > $OUT/logs/$arm-E$E-s$s.log 2>&1
  echo "done $arm $E $s rc=$? $(date -u +%T)"
}
export -f one
cat grid.txt | xargs -P 7 -L 1 bash -c 'one $0 $1 $2' | tee $OUT/progress.txt
ls $OUT/runs/*.json | wc -l
