#!/bin/bash
# M7 w4: 36 runs, 8 parallel CPU processes on the VM.
set -u
OUT=${OUT:-/root/out}
PY=${PY:-/opt/tn/bin/python}
mkdir -p $OUT/results
jobs_list=()
for E in 8 64; do for s in 1 2 3; do
  jobs_list+=("none 0 $E $s" "deepseek 0 $E $s" "global 100 $E $s" "global 1000 $E $s" "perfam 100 $E $s" "perfam 1000 $E $s")
done; done
for j in "${jobs_list[@]}"; do
  set -- $j
  name="$1-b$2-E$3-s$4"
  echo "$name"
done | xargs -P 8 -I{} bash -c 'n={}; IFS=- read arm b E s <<< "$n"; OMP_NUM_THREADS=1 timeout 3600 '"$PY"' moe_w4.py --arm $arm --beta ${b#b} --E ${E#E} --seed ${s#s} --out '"$OUT"'/results/$n > '"$OUT"'/results/$n.log 2>&1; echo done $n rc=$?'

$PY analyze_w4.py $OUT/results > $OUT/analyze.txt 2>&1
echo ALL DONE
