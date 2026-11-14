#!/bin/bash
# Throttled local runner for the w4 grid: starts a run only while fleet study python jobs < 12
# and at most 4 of mine run; skips runs whose info.json exists (local or GPU-synced).
J=/home/agent/workspaces/eurisko-study/traders-net/study/M7/w4/job
OUT=/tmp/traders-net/M7w4/local
mkdir -p $OUT
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1
count_study() { for p in $(pgrep -x python3); do a=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null); c=$(readlink /proc/$p/cwd 2>/dev/null); case "$a $c" in *grafana*) ;; *eurisko-study*) echo x;; esac; done | wc -l; }
mine() { pgrep -f "moe_w4.py" | wc -l; }
list=()
for s in 1 2 3; do for E in 8 64; do
  list+=("deepseek 0 $E $s" "global 100 $E $s" "global 1000 $E $s" "none 0 $E $s" "perfam 100 $E $s" "perfam 1000 $E $s")
done; done
for j in "${list[@]}"; do
  set -- $j; n="$1-b$2-E$3-s$4"
  [ -f $OUT/$n/info.json ] && continue
  [ -f /home/agent/workspaces/eurisko-study/traders-net/study/M7/w4/gpu/results/$n/info.json ] && continue
  while [ $(count_study) -ge 12 ] || [ $(mine) -ge 4 ]; do sleep 20; done
  echo "$(date -u +%T) start $n"
  (cd $J && timeout 5400 python3 moe_w4.py --arm $1 --beta $2 --E $3 --seed $4 --out $OUT/$n > $OUT/$n.log 2>&1; echo "$(date -u +%T) end $n rc=$?") &
  sleep 30
done
wait
echo ALL-LOCAL-DONE
