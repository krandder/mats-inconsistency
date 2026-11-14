cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python; SM=/tmp/e56_smoke; mkdir -p $SM
C="--const 3000 --cooldown 0 --ckpt dense --seed 0 --pair_init"
$PY train_e56.py --f0 sep --f1 sep $C --out $SM/e56_sep_s0 > $SM/e56_sep_s0.log 2>&1
$PY train_e56.py --variant unif --f0 sep --f1 sep $C --out $SM/e56_unifsep_s0 > $SM/e56_unifsep_s0.log 2>&1
$PY e56_eval.py $SM $SM/res 0 10 > $SM/eval.log 2>&1
$PY e56_plumb.py > $SM/plumb.out 2>&1
touch $SM/done
