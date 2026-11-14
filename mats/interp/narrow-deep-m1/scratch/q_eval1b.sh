#!/bin/bash
# Eval queue part 1b: for nets whose samplers already ran: k-by-cut oracle and forward study only.
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
NAME=$1; shift
for spec in "$@"; do set -- $spec; S=$1; G=$2
  T=runs/truth_s${S}_sig${G}.json
  $N9 $PY scratch/oracle_k_by_cut.py $S $G runs/kbycut_s${S}_sig${G}.json 15 >> logs/kbycut_$NAME.log 2>&1
  $N9 $PY scratch/fwd_study.py $S $G $T runs/fwd_s${S}_sig${G}.json 1e-3,1e-4 2,4,8 >> logs/fwd_$NAME.log 2>&1
done
echo DONE > logs/q_eval1b_$NAME.done
