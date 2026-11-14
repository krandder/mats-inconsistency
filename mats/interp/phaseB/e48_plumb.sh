#!/usr/bin/env bash
# E48 plumbing runs (CPU, 500 steps each, nice 19; disclosed in amendment 56; not results). Then e48_plumb.py compares them.
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
PL=/tmp/e48_plumb
mkdir -p $PL
export OMP_NUM_THREADS=1
C="--const 500 --cooldown 0 --ckpt dense --seed 0"
run() { nice -n 19 ionice -c3 $PY "$@" > $PL/$(basename ${@: -1}).log 2>&1; }
run train_sw.py --f0 ctrl --f1 ctrl $C --out $PL/w1_sw_ctrl &
run train_lin.py --f0 ctrl --f1 ctrl $C --out $PL/w1_lin_ctrl &
run train_lin.py --f0 sq --f1 sq $C --out $PL/w1_sq &
run train_lin.py --f0 sgnsq --f1 sgnsq --pair_init $C --out $PL/w1_sgnsq &
wait
run train_t2.py --f0 ctrl --f1 ctrl --D 2 $C --out $PL/w2_t2_ctrl &
run train_t2lin.py --fmt ctrl --D 2 $C --out $PL/w2_lin_ctrl &
run train_t2lin.py --fmt prod --D 2 $C --out $PL/w2_prod &
run train_t2lin.py --fmt sgnprod --D 2 $C --out $PL/w2_sgnprod &
wait
nice -n 19 ionice -c3 $PY e48_plumb.py > $PL/plumb.out 2>&1
touch $PL/done
