#!/bin/bash
# M6 stress s2: 72 prequential CPU runs (8 arms x {clean,iid,fixed} x 3 seeds), 8 in parallel.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import torch,numpy;print(torch.__version__, numpy.__version__)"
jobs=""
for nz in iid clean fixed; do for s in 1 2 3; do for a in a b cg cq fb01 fb05 wd boost; do jobs="$jobs $a:$nz:$s"; done; done; done
echo $jobs | tr ' ' '\n' | xargs -P 8 -I{} sh -c 'IFS=:; set -- {}; [ -f $OUT/$1-$2-s$3.json ] || python noise.py --arm $1 --noise $2 --seed $3 --out $OUT > $OUT/log-$1-$2-s$3.txt 2>&1; echo done {} $?'
ls $OUT/*.json | wc -l
