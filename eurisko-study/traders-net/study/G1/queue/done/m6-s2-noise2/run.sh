#!/bin/bash
# M6 stress s2: 72 prequential CPU runs (8 arms x {iid,clean,fixed} x 3 seeds), 8 in parallel.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"; cp -n prior/*.json "$OUT"/ 2>/dev/null; echo "prior json: $(ls prior 2>/dev/null | wc -l)"
python -c "import torch,numpy;print(torch.__version__, numpy.__version__)"
for nz in iid clean fixed; do for s in 1 2 3; do for a in a b cg cq fb01 fb05 wd boost; do echo "$a $nz $s"; done; done; done > jobs.txt
xargs -P 8 -L 1 bash -c '[ -f "$OUT/$0-$1-s$2.json" ] || python noise.py --arm "$0" --noise "$1" --seed "$2" --out "$OUT" > "$OUT/log-$0-$1-s$2.txt" 2>&1; echo "done $0 $1 $2 rc=$?"' < jobs.txt
n=$(ls $OUT/*.json | wc -l); echo "json=$n"; [ "$n" -ge 72 ]
