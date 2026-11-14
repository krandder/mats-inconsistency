#!/bin/bash
# M6 s1 stress: 2 regimes x 8 arms x 3 seeds = 48 runs on one L4 VM, 12 at a time (CUDA, small models).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import torch,numpy;print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
jobs=""
for s in 1 2 3; do for r in IE MS; do for a in A BOOST WD B CG CQ FB01 FB05; do jobs="$jobs $r:$a:$s"; done; done; done
echo $jobs | tr ' ' '\n' | xargs -P 12 -I{} sh -c 'IFS=:; set -- {}; [ -f $OUT/$1-$2-s$3.json ] || python run_s1.py $1 $2 $3 --out $OUT > $OUT/log-$1-$2-s$3.txt 2>&1; echo done {} $?'
ls $OUT/*.json | wc -l
