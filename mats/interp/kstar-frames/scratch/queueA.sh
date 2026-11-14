#!/bin/bash
# Run A queue: 18 configs (3 widths x 3 seeds x 2 depths), 3 at a time, nice 19 ionice -c3, 3 threads each.
cd /home/agent/workspaces/mats/work/kstar-frames/scratch
PY=/home/agent/.venvs/arc-noisy/bin/python
jobs=""
for n in 128 64 32; do for L in 32 16; do for s in 400 401 402; do jobs="$jobs $n:$s:$L"; done; done; done
echo $jobs | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'IFS=: read n s L <<< "{}"; OMP_NUM_THREADS=3 nice -n 19 ionice -c3 '$PY' r1_layers.py --n $n --seed $s --L $L --threads 3 > ../logs/A_n${n}_s${s}_L${L}.log 2>&1'
echo "queueA finished $(date -u)" >> ../logs/queueA.done
