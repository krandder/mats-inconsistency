#!/bin/bash
# Per-net queue: wait for the truth, then samplers, k-by-cut oracle (2^15), deterministic arms; then extra truths.
# usage: q_net.sh NAME SEED SIG -- "TRUTHSPEC" ...
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
NAME=$1; S=$2; G=$3; shift 4
T=runs/truth_s${S}_sig${G}.json
while [ ! -f $T ]; do sleep 30; done
$N9 $PY scratch/samplers.py $S $G $T runs/samp_s${S}_sig${G}.json 1 20 >> logs/samp_$NAME.log 2>&1
$N9 $PY scratch/oracle_k_by_cut.py $S $G runs/kbycut_s${S}_sig${G}.json 15 >> logs/kbycut_$NAME.log 2>&1
scratch/q_eval2.sh ${NAME}e "$S $G"
for spec in "$@"; do set -- $spec
  $N9 $PY scratch/truth.py $1 $2 runs/truth_s$1_sig$2 $3 >> logs/truth_$NAME.log 2>&1
done
echo DONE > logs/q_net_$NAME.done
