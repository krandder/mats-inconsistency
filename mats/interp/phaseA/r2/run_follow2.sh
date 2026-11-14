#!/bin/bash
# follow-up queue 2: QT pool, then matched-QT oracle (r2qt) and readout control (r2loc); total round-2 python <= 4
cd /home/agent/workspaces/mats/work/cm-interp
export OMP_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
nrun() { pgrep -fc "^/home/agent/.venvs/aggext/bin/python (r2run|r2loc|r2qt|r2qtpool|r2qtinfo|r2summ).py" ; }
while [ "$(nrun)" -ge 4 ]; do sleep 15; done
$PY r2qtpool.py > r2/logs/qtpool.log 2>&1; echo "qtpool exit $?" >> r2/logs/run_follow.status
JOBS="qt:2L_s0 qt:2L_s1 loc:2L_s1 qt:3L_s10 qt:3L_s11 loc:2L_s2 qt:2L_s2 qt:2L_s3 loc:2L_s3 qt:3L_s12 qt:3L_s13 loc:3L_s10 qt:2L_s4 qt:3L_s14 qt:plat_s3 loc:3L_s11 loc:2L_s4 loc:plat_s3 loc:3L_s12 loc:3L_s13 loc:3L_s14"
for j in $JOBS; do
  kind=${j%%:*}; net=${j##*:}
  while [ "$(nrun)" -ge 4 ]; do sleep 15; done
  if [ "$kind" = qt ]; then
    while [ ! -f r2/$net.json ] || ! grep -q js_binding r2/$net.json; do sleep 30; done   # needs r2run finished? no: independent, but keep order
    (setsid $PY r2qt.py $net > r2/logs/qt_$net.log 2>&1; echo "qt $net exit $?" >> r2/logs/run_follow.status) &
  else
    (setsid $PY r2loc.py $net > r2/logs/loc_$net.log 2>&1; echo "loc $net exit $?" >> r2/logs/run_follow.status) &
  fi
  sleep 8
done
wait
