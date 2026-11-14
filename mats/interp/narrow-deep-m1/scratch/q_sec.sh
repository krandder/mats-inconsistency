#!/bin/bash
# Secondary arms: samplers (no HYB), BO, TS (GHM 1e-3), deep-noise diagnostic. usage: q_sec.sh NAME "SEED SIG" ...
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
NAME=$1; shift
for spec in "$@"; do set -- $spec; S=$1; G=$2
  T=runs/truth_s${S}_sig${G}.json
  if [ ! -f $T ]; then $N9 $PY scratch/truth.py $S $G runs/truth_s${S}_sig${G} 23 >> logs/truth_$NAME.log 2>&1; fi
  $N9 $PY scratch/samplers.py $S $G $T runs/samp_s${S}_sig${G}.json 0 20 >> logs/samp_$NAME.log 2>&1
  $N9 $PY scratch/bo_run.py $S $G $T runs/bo_s${S}_sig${G}.json 64 2,3,4,5,6 4,8 >> logs/bo_$NAME.log 2>&1
  $N9 $PY scratch/ts_run.py $S $G $T runs/ts_s${S}_sig${G}.json "GHM,2,1e-3,4,0,0,32;GHM,4,1e-3,4,0,0,32;K2X,2,0,4,0,0,32" >> logs/ts_$NAME.log 2>&1
  $N9 $PY scratch/deep_noise_effect.py $S $G 32 >> logs/dne_$NAME.log 2>&1
done
echo DONE > logs/q_sec_$NAME.done
