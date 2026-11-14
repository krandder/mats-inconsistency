#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
for spec in "L2ws0_s0.05 /home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s0 5" "L2ws3_s0.05 /home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s3 3" "L2mx0_s0.05 /home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_s0 6" "L2mx1_s0.05 /home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_s1 7" "L2mx4_s0.05 /home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_s4 8"; do
  set -- $spec
  ./runwhen.sh $PY nnoise.py train --init $2 --sigma 0.05 --seed $3 --steps 20000 --cooldown 8000 --ramp 1000 --out runs/$1 > runs/$1.log 2>&1 &
  sleep 30
done
wait
