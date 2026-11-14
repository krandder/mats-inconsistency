#!/usr/bin/env bash
# E49 after the VM (fifth agent): collect the VM's runs into runs/e49/, link world 1 and world 2 runs into separate directories,
# evaluate through the CPU queue (4 processes, nice 19) with e49_eval.py, then the summary.
cd /home/agent/workspaces/mats/work/cm-interp-B
./gpu/collect_b4.sh cmb-e49-1
mkdir -p runs/e49w1 runs/e49t2
for d in runs/e49/e49_w1*; do [ -d "$d" ] && ln -sfn ../e49/$(basename $d) runs/e49w1/$(basename $d); done
for d in runs/e49/e49_d*; do [ -d "$d" ] && ln -sfn ../e49/$(basename $d) runs/e49t2/$(basename $d); done
echo "w1 runs: $(ls runs/e49w1/*/meta.json | wc -l); t2 runs: $(ls runs/e49t2/*/meta.json | wc -l)"
PY=/home/agent/.venvs/aggext/bin/python
$PY mkq.py "e49_eval.py w1" runs/e49w1 results/e49w1 2 /tmp/q_e49.txt
$PY mkq.py "e49_eval.py t2" runs/e49t2 results/e49t2 4 /tmp/q_e49.txt append
./cpuq.sh /tmp/q_e49.txt results/e49_eval.log 4
nice -n 19 ionice -c3 $PY e49_summ.py > results/e49_summ.out 2>&1
touch /tmp/e49_done
