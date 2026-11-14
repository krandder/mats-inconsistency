#!/usr/bin/env bash
# analysis queue 2: eval (with noiseless-agreement KL) for every finished run that lacks it; MI + xg + e3 for the
# analysis candidates. Waits for a free slot (lane total <= 4 active python processes).
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
active() { ps -eo stat,args | grep -E "python[^ ]* (-W ignore )?(nnoise|mirun|task1|task3|e1|e2|e3|pfn_rare|pfn_diag|dec|twist)\.py" | grep -v grep | grep -v "bash " | awk '$1 !~ /T/' | wc -l; }
slot() { while [ $(active) -ge 4 ]; do sleep 30; done; }
for pass in $(seq 1 400); do
  pending=0
  for r in runs/*/; do
    r=${r%/}; n=$(basename $r)
    [ -e $r/meta.json ] || { pending=1; continue; }
    if [ ! -e $r/eval.json ] || ! grep -q agree_kl $r/eval.json; then
      case $n in *_s0.0) [ -e $r/eval.json ] && continue;; esac
      pgrep -f "nnoise.py eval --run $r\$" > /dev/null && continue
      slot; nice -n 5 $PY nnoise.py eval --run $r > $r/eval.log 2>&1
    fi
  done
  for n in L2_s0.05 L3_s0.05 L2_s0.0316 L2_s0.183 L2w64_s0.183 L2_s0.1; do
    r=runs/$n
    [ -e $r/meta.json ] || continue
    [ -e $r/mi.done ] || { slot; nice -n 5 $PY mirun.py $r > $r/mi.log 2>&1 && touch $r/mi.done; }
    [ -e $r/mi_xg.json ] || { slot; MIRUN_XG=1 nice -n 5 $PY mirun.py $r > $r/mi_xg.log 2>&1; }
    [ -e $r/e3.json ] || { slot; nice -n 5 $PY e3.py $r > $r/e3.log 2>&1; }
  done
  [ $pending = 0 ] && ! pgrep -f "runq_c.sh" > /dev/null && break
  sleep 120
done
echo ANQ2-DONE
