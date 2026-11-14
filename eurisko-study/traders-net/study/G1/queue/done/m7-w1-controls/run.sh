#!/bin/bash
# M7 w1 Stage-1 controls: 33 runs, 6 in parallel on one L4.
set -u
export M7_SUBSTRATE=$PWD/substrate
mkdir -p $OUT/runs $OUT/logs
nvidia-smi -L > $OUT/gpu.txt 2>&1
one() {
  arm=$1; E=$2; s=$3
  case $arm in
    deepseek-u1e-3) extra="--arm deepseek --u 1e-3 --tag $arm";;
    deepseek-u1e-2) extra="--arm deepseek --u 1e-2 --tag $arm";;
    switch) extra="--arm switch --alpha 1e-2";;
    *) extra="--arm $arm";;
  esac
  python moe_harness.py $extra --E $E --seed $s --T 24000 --B 8 --device cuda --threads 1 --out $OUT/runs > $OUT/logs/$arm-E$E-s$s.log 2>&1
  echo "done $arm $E $s rc=$?"
}
export -f one
cat grid.txt | xargs -P 6 -L 1 bash -c 'one $0 $1 $2'
ls $OUT/runs/*.json | wc -l
