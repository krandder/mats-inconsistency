#!/usr/bin/env bash
# Scheduler (amendment 2): width arm first, then the low-sigma runs; keeps this lane's active python processes <= 4
# (stopped processes do not count). Finally resumes the two paused sigma = 0.8 runs.
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
I2=/home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s4
I3=/home/agent/workspaces/mats/work/tiny-pfn-interp/heads/hruns/L3_ws_s13
active() { ps -eo stat,args | grep -E "python[^ ]* (-W ignore )?(nnoise|mirun|task1|task3|e1|e2|e3|pfn_rare|pfn_diag|dec|twist)\.py" | grep -v grep | grep -v "bash " | awk '$1 !~ /T/' | wc -l; }
launch() {   # name init sigma width
  [ -e runs/$1.log ] && return
  while [ $(active) -ge 4 ]; do sleep 30; done
  nohup nice -n 5 $PY nnoise.py train --init $2 --sigma $3 --width $4 --steps 20000 --cooldown 8000 --ramp 1000 --out runs/$1 > runs/$1.log 2>&1 &
  sleep 20
}
launch L2_s0.183 $I2 0.183 32
launch L2w48_s0.183 $I2 0.183 48
launch L2w64_s0.183 $I2 0.183 64
launch L2_s0.316 $I2 0.316 32
launch L2w48_s0.316 $I2 0.316 48
launch L2w64_s0.316 $I2 0.316 64
launch L2_s0.05 $I2 0.05 32
launch L3_s0.05 $I3 0.05 32
launch L2_s0.0316 $I2 0.0316 32
launch L2_s0.0577 $I2 0.0577 32
launch L3_s0.0316 $I3 0.0316 32
launch L3_s0.0577 $I3 0.0577 32
for p in $(ps -eo pid,stat,args | grep "nnoise.py train" | grep -v grep | awk '$2 ~ /T/ {print $1}'); do
  while [ $(active) -ge 4 ]; do sleep 30; done
  kill -CONT $p; sleep 5
done
wait
echo QUEUE-C-DONE
