#!/bin/bash
# Eval queue part 2 (deterministic arms + practical arm + deep-noise diagnostic). usage: q_eval2.sh NAME "SEED SIG" ...
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
NAME=$1; shift
LAD="GHM,2,1e-3,4,0,0,32;GHM,4,1e-3,4,0,0,32;GHM,8,1e-3,4,0,0,32;GHM,2,1e-4,4,0,0,32;GHM,4,1e-4,4,0,0,32;GHM,8,1e-4,4,0,0,32;K2X,2,0,4,0,0,32;K2X,4,0,4,0,0,32;K2X,8,0,4,0,0,32"
for spec in "$@"; do set -- $spec; S=$1; G=$2
  T=runs/truth_s${S}_sig${G}.json
  while [ ! -f $T ]; do sleep 30; done
  $N9 $PY scratch/bo_run.py $S $G $T runs/bo_s${S}_sig${G}.json 64 2,3,4,5,6 4,8,16 >> logs/bo_$NAME.log 2>&1
  $N9 $PY scratch/deep_noise_effect.py $S $G 32 >> logs/dne_$NAME.log 2>&1
  if [ "$G" == "0.001" ]; then L2=$LAD; else L2="GHM,2,1e-3,4,0,0,32;GHM,4,1e-3,4,0,0,32;GHM,8,1e-3,4,0,0,32"; fi
  $N9 $PY scratch/ts_run.py $S $G $T runs/ts_s${S}_sig${G}.json "$L2" >> logs/ts_$NAME.log 2>&1
  $N9 $PY scratch/sh_run.py $S $G $T runs/sh_s${S}_sig${G}.json 8 32 4 >> logs/sh_$NAME.log 2>&1
done
echo DONE > logs/q_eval2_$NAME.done
