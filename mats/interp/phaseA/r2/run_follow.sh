#!/bin/bash
# follow-up queue: QT info check + readout-control localisation; keeps total round-2 python processes <= 4
cd /home/agent/workspaces/mats/work/cm-interp
export OMP_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
nrun() { pgrep -fc "^/home/agent/.venvs/aggext/bin/python (r2run|r2loc|r2qtinfo).py" ; }
JOBS="qtinfo 2L_s0 2L_s1 2L_s2 2L_s3 2L_s4 plat_s3 3L_s10 3L_s11 3L_s12 3L_s13 3L_s14"
for j in $JOBS; do
  while [ "$(nrun)" -ge 4 ]; do sleep 20; done
  if [ "$j" = qtinfo ]; then
    (setsid $PY r2qtinfo.py > r2/logs/qtinfo.log 2>&1; echo "qtinfo exit $?" >> r2/logs/run_follow.status) &
  else
    (setsid $PY r2loc.py $j > r2/logs/loc_$j.log 2>&1; echo "loc $j exit $?" >> r2/logs/run_follow.status) &
  fi
  sleep 5
done
wait
