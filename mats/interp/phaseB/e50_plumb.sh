#!/usr/bin/env bash
# E50 plumbing and smoke runs (CPU, seed 0, 3,000 constant-lr steps, nice 19; disclosed in amendment 58; not results).
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
SM=/tmp/e50_smoke
mkdir -p $SM
export OMP_NUM_THREADS=1
C="--const 3000 --cooldown 0 --ckpt dense --seed 0"
run() { nice -n 19 ionice -c3 $PY "$@" > $SM/$(basename ${@: -1}).log 2>&1; }
run train_noln.py --f0 ctrl --f1 ctrl $C --out $SM/e50_ctrl_s0 &
run train_noln.py --f0 mag --f1 mag $C --out $SM/e50_mag_s0 &
run train_noln.py --f0 sq --f1 sq $C --out $SM/e50_sq_s0 &
wait
nice -n 19 ionice -c3 $PY e50_eval.py $SM $SM/res 0 10 > $SM/eval.log 2>&1
nice -n 19 ionice -c3 $PY e50_plumb.py > $SM/plumb.out 2>&1
(cd /tmp/e50_synth && nice -n 19 $PY /home/agent/workspaces/mats/work/cm-interp-B/e50_summ.py > /tmp/e50_synth/summ.out 2>&1)
touch $SM/done
