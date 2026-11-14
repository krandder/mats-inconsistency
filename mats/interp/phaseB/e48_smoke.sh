#!/usr/bin/env bash
# E48 smoke runs (CPU, seed 0, 3,000 constant-lr steps, nice 19; disclosed in amendment 56; not results): the new formats train
# without numerical trouble, and the evaluators run on them.
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
SM=/tmp/e48_smoke
mkdir -p $SM
export OMP_NUM_THREADS=1
C="--const 3000 --cooldown 0 --ckpt none --seed 0"
run() { nice -n 19 ionice -c3 $PY "$@" > $SM/$(basename ${@: -1}).log 2>&1; }
run train_lin.py --f0 sq --f1 sq $C --out $SM/w1/e48_w1sq_s0 &
run train_lin.py --f0 sgnsq --f1 sgnsq --pair_init $C --out $SM/w1/e48_w1sgnsq_s0 &
run train_t2lin.py --fmt prod --D 2 $C --out $SM/w2/e48_d2prod_s0 &
run train_t2lin.py --fmt sgnprod --D 2 $C --out $SM/w2/e48_d2sgnprod_s0 &
wait
run train_t2lin.py --fmt prod --D 1 $C --out $SM/w2/e48_d1prod_s0 &
run train_t2lin.py --fmt sgnprod --D 1 $C --out $SM/w2/e48_d1sgnprod_s0 &
wait
nice -n 19 ionice -c3 $PY e48_eval_w1.py $SM/w1 $SM/res_w1 0 10 > $SM/eval_w1.log 2>&1 &
nice -n 19 ionice -c3 $PY e48_eval_t2.py $SM/w2 $SM/res_t2 0 10 > $SM/eval_t2.log 2>&1 &
wait
touch $SM/done
