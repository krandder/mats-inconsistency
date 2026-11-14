#!/bin/bash
# M7 w5 stage 1: bankruptcy+birth (split, fresh) + same-VM none/DeepSeek comparators. 30 runs, 6 parallel.
set -u
export M7_SUBSTRATE=$PWD/substrate M7_HARNESS_DIR=$PWD
mkdir -p $OUT/runs $OUT/logs
nvidia-smi -L > $OUT/gpu.txt 2>&1
cp SHA256SUMS $OUT/ 2>/dev/null
one() {
  arm=$1; E=$2; s=$3
  case $arm in
    deepseek-u1e-3) extra="--arm deepseek --u 1e-3 --tag $arm";;
    deepseek-u1e-2) extra="--arm deepseek --u 1e-2 --tag $arm";;
    *) extra="--arm $arm";;
  esac
  python bb.py $extra --E $E --seed $s --T 24000 --B 8 --H 200 --device cuda --threads 1 --out $OUT/runs > $OUT/logs/$arm-E$E-s$s.log 2>&1
  echo "done $arm $E $s rc=$? $(date -u +%T)"
}
export -f one
cat grid.txt | xargs -P 6 -L 1 bash -c 'one $0 $1 $2'
ls $OUT/runs/*.json | wc -l
