#!/bin/bash
# 24 prequential CPU runs (8 arms x 3 seeds), at most 3 in parallel; skips finished runs.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
mkdir -p results
for s in 1 2 3; do for a in a b cg cq fb01 fb05 wd boost; do
  [ -f "results/li-$a-s$s.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  python3 dfam.py "$a" "$s" >> results/run.log 2>&1 &
done; done
wait
ls results/*.json | wc -l
