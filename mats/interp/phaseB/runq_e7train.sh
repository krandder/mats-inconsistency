#!/usr/bin/env bash
# E7.2: three new from-scratch bases (tiny-pfn recipe) + noise fine-tunes; plus the unused base mix4_s2
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python; TPF=/home/agent/workspaces/mats/work/tiny-pfn
./runwhen.sh $PY nnoise.py train --init $TPF/runs/w32_h2_L2_mix4_s2 --sigma 0.05 --seed 9 --steps 20000 --cooldown 8000 --ramp 1000 --out runs/L2mx2_s0.05 > runs/L2mx2_s0.05.log 2>&1 &
sleep 20
for s in 5 6 7; do
  ( ./runwhen.sh $PY $TPF/pfn.py train --width 32 --heads 2 --depth 2 --mixk 4 --seed $s --out runs/base_mix4_s$s > runs/base_mix4_s$s.log 2>&1
    ./runwhen.sh $PY nnoise.py train --init runs/base_mix4_s$s --sigma 0.05 --seed $((10 + s)) --steps 20000 --cooldown 8000 --ramp 1000 --out runs/L2nb${s}_s0.05 > runs/L2nb${s}_s0.05.log 2>&1 ) &
  sleep 20
done
wait
