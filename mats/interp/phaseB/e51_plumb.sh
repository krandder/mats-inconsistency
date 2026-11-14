#!/usr/bin/env bash
# E51 plumbing (CPU, nice 19; disclosed in amendment 59; not results): the run script on 60 episodes per setting into a scratch
# directory, the summary on those outputs, and the synthetic checks of the statistics.
cd /home/agent/workspaces/mats/work/cm-interp-B
PY=/home/agent/.venvs/aggext/bin/python
PL=/tmp/e51_plumb
mkdir -p $PL/results/e51
export OMP_NUM_THREADS=1 E51_NB=60 E51_NF=60 E51_OUT=$PL/results/e51
for st in w1 d1 d2; do nice -n 19 ionice -c3 $PY e51_run.py $st > $PL/run_$st.log 2>&1 & done
wait
(cd $PL && nice -n 19 $PY /home/agent/workspaces/mats/work/cm-interp-B/e51_summ.py > $PL/summ.out 2>&1)
unset E51_NB E51_NF E51_OUT
nice -n 19 $PY e51_plumb.py > $PL/synth.out 2>&1
touch $PL/done
