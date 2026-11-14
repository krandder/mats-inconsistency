#!/usr/bin/env bash
# E48 after the VMs (fifth agent): collect the three VMs' runs into runs/e48/, link world 1 and world 2 runs into separate
# directories for the evaluators, and evaluate through the CPU queue (4 processes, nice 19). Then summary and post hoc check.
cd /home/agent/workspaces/mats/work/cm-interp-B
for r in cmb-e48a-1 cmb-e48b-1 cmb-e48c-1; do ./gpu/collect_b4.sh $r; done
mkdir -p runs/e48w1 runs/e48t2
for d in runs/e48/e48_w1*; do [ -d "$d" ] && ln -sfn ../e48/$(basename $d) runs/e48w1/$(basename $d); done
for d in runs/e48/e48_d*; do [ -d "$d" ] && ln -sfn ../e48/$(basename $d) runs/e48t2/$(basename $d); done
echo "w1 runs: $(ls runs/e48w1/*/meta.json | wc -l); t2 runs: $(ls runs/e48t2/*/meta.json | wc -l)"
PY=/home/agent/.venvs/aggext/bin/python
$PY mkq.py e48_eval_w1.py runs/e48w1 results/e48w1 2 /tmp/q_e48.txt
$PY mkq.py e48_eval_t2.py runs/e48t2 results/e48t2 4 /tmp/q_e48.txt append
./cpuq.sh /tmp/q_e48.txt results/e48_eval.log 4
nice -n 19 ionice -c3 $PY e48_summ.py > results/e48_summ.out 2>&1
OMP_NUM_THREADS=1 nice -n 19 ionice -c3 $PY e48_post.py > results/e48_post.out 2>&1
touch /tmp/e48_done
