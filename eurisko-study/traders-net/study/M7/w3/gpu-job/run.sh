#!/bin/bash
# M7 w3 stage 1: exp3 shadow credit + fullinfo diagnostic + same-code none/deepseek, E 8/64, 3 seeds.
set -u
export M7_SUBSTRATE=$PWD/substrate
mkdir -p $OUT/runs $OUT/logs
nvidia-smi -L > $OUT/gpu.txt 2>&1
one() {
  arm=$1; E=$2; s=$3
  python exp3.py --arm $arm --E $E --seed $s --T 24000 --B 8 --device cuda --threads 1 --out $OUT/runs > $OUT/logs/$arm-E$E-s$s.log 2>&1
  echo "done $arm $E $s rc=$?"
}
export -f one
cat grid.txt | xargs -P 7 -L 1 bash -c 'one $0 $1 $2'
ls $OUT/runs/*.json | wc -l
