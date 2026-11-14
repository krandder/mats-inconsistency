#!/usr/bin/env bash
# E53 plumbing and smoke runs (CPU, seed 0, 3,000 constant-lr steps, nice 19; disclosed in amendment 61; not results).
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
SM=/tmp/e53_smoke
mkdir -p $SM
export OMP_NUM_THREADS=1
C="--const 3000 --cooldown 0 --ckpt dense --seed 0"
run() { nice -n 19 ionice -c3 $PY "$@" > $SM/$(basename ${@: -1}).log 2>&1; }
run train_uni.py --variant uni --f0 ctrl --f1 ctrl $C --out $SM/e53_uni_s0 &
run train_uni.py --variant unif --f0 ctrl --f1 ctrl $C --out $SM/e53_unif_s0 &
run train_uni.py --variant unif_noln --f0 mag --f1 mag $C --out $SM/e53_unifnoln_mag_s0 &
run train_uni.py --variant unif_noln --f0 sq --f1 sq $C --out $SM/e53_unifnoln_sq_s0 &
wait
nice -n 19 ionice -c3 $PY e53_eval.py $SM $SM/res 0 10 > $SM/eval.log 2>&1
nice -n 19 ionice -c3 $PY e53_plumb.py > $SM/plumb.out 2>&1
mkdir -p /tmp/e53_synth
(cd /tmp/e53_synth && nice -n 19 $PY /home/agent/workspaces/mats/work/cm-interp-B/e53_summ.py > /tmp/e53_synth/summ.out 2>&1)
touch $SM/done
