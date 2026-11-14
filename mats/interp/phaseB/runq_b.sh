#!/usr/bin/env bash
# second queue (amendment 1): SNR 1000, 300, 10, 3 -> sigma 0.0316, 0.0577, 0.316, 0.577; shares the 2-slot training limit
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
I2=/home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s4
I3=/home/agent/workspaces/mats/work/tiny-pfn-interp/heads/hruns/L3_ws_s13
sleep 10
for j in L2:0.0316 L3:0.0316 L2:0.0577 L3:0.0577 L2:0.316 L2:0.577; do
  m=${j%%:*}; s=${j##*:}; init=$I2; [ $m = L3 ] && init=$I3
  [ -e runs/${m}_s$s.log ] && continue
  while [ $(pgrep -f "python nnoise.py train" | wc -l) -ge 2 ] || pgrep -f "runq.sh" > /dev/null; do sleep 23; done
  nohup nice -n 5 $PY nnoise.py train --init $init --sigma $s --steps 20000 --cooldown 8000 --ramp 1000 --out runs/${m}_s$s > runs/${m}_s$s.log 2>&1 &
  sleep 5
done
wait
echo QUEUE-B-DONE
