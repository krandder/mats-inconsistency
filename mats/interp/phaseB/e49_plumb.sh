#!/usr/bin/env bash
# E49 plumbing and smoke runs (CPU, seed 0, 3,000 constant-lr steps, nice 19; disclosed in amendment 57; not results).
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
SM=/tmp/e49_smoke
mkdir -p $SM/w1 $SM/t2
export OMP_NUM_THREADS=1
C="--f0 ctrl --f1 ctrl --const 3000 --cooldown 0 --ckpt dense --seed 0"
run() { nice -n 19 ionice -c3 $PY "$@" > $SM/$(basename ${@: -1}).log 2>&1; }
run train_pre.py w1 $C --out $SM/w1/e49_w1pre_s0 &
run train_pre.py t2 $C --D 2 --out $SM/t2/e49_d2pre_s0 &
run train_pre.py t2 $C --D 1 --out $SM/t2/e49_d1pre_s0 &
wait
ln -sfn $SM/w1/e49_w1pre_s0 $SM/e49_w1pre_s0; ln -sfn $SM/t2/e49_d2pre_s0 $SM/e49_d2pre_s0
nice -n 19 ionice -c3 $PY e49_eval.py w1 $SM/w1 $SM/res_w1 0 10 > $SM/eval_w1.log 2>&1 &
nice -n 19 ionice -c3 $PY e49_eval.py t2 $SM/t2 $SM/res_t2 0 10 > $SM/eval_t2.log 2>&1 &
wait
nice -n 19 ionice -c3 $PY e49_plumb.py > $SM/plumb.out 2>&1
(cd /tmp/e49_synth && nice -n 19 $PY /home/agent/workspaces/mats/work/cm-interp-B/e49_summ.py > /tmp/e49_synth/summ.out 2>&1)
touch $SM/done
