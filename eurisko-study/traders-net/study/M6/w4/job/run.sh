#!/bin/bash
# M6 w4: 36 prequential CPU runs (2 streams x 6 arms x 3 seeds), 8 in parallel on the VM's vCPUs.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import torch,numpy;print(torch.__version__, numpy.__version__)"
python test_fp.py || exit 3
jobs=""
for st in li base; do for s in 1 2 3; do for a in fp pass2 pass1 plain plain_wd boost; do jobs="$jobs $st:$a:$s"; done; done; done
echo $jobs | tr ' ' '\n' | xargs -P 8 -I{} sh -c 'IFS=:; set -- {}; [ -f $OUT/$1-$2-s$3.json ] || python fp_layer.py $1 $2 $3 > $OUT/log-$1-$2-s$3.txt 2>&1; echo done {} $?'
ls $OUT/*.json | wc -l
