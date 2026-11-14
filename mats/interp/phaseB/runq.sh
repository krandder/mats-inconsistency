#!/usr/bin/env bash
# queue: noisy fine-tunes; at most 3 concurrent training jobs (leaves 1 of the lane's 4 CPU slots for analysis)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
I2=/home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s4
I3=/home/agent/workspaces/mats/work/tiny-pfn-interp/heads/hruns/L3_ws_s13
jobs_list=""
for s in 0.4 1.6 0.1 0.8 0.05 0.2; do jobs_list="$jobs_list L3:$s L2:$s"; done
for j in $jobs_list; do
  m=${j%%:*}; s=${j##*:}; init=$I2; [ $m = L3 ] && init=$I3
  [ -e runs/${m}_s$s.log ] && continue
  while [ $(pgrep -f "python nnoise.py train" | wc -l) -ge 2 ]; do sleep 20; done
  nohup nice -n 5 $PY nnoise.py train --init $init --sigma $s --steps 20000 --cooldown 8000 --ramp 1000 --out runs/${m}_s$s > runs/${m}_s$s.log 2>&1 &
  sleep 5
done
wait
echo QUEUE-DONE
