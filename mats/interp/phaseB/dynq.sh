#!/usr/bin/env bash
# Priority CPU queue (sixth agent). Jobs are files in DIR/pending, one command line each (run from cm-interp-B); the runner
# starts the lexicographically first pending job whenever fewer than MAXP heavy processes run, counting its own jobs and any
# legacy PIDs listed in DIR/legacy_pids (jobs started before this queue). Each job runs at the lowest CPU and I/O priority with
# OMP_NUM_THREADS=1. A job can be added at any time (lower name = higher priority). The runner exits when DIR/stop exists and
# nothing is pending or running. Log: LOG (appended).
# usage: dynq.sh DIR LOG [MAXP]
cd /home/agent/workspaces/mats/work/cm-interp-B
D=$1; LOG=$2; MAXP=${3:-4}
mkdir -p "$D/pending" "$D/running" "$D/done"
export OMP_NUM_THREADS=1
legacy() { [ -f "$D/legacy_pids" ] || { echo 0; return; }; n=0; for p in $(cat "$D/legacy_pids"); do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
busy() { echo $(( $(jobs -rp | wc -l) + $(legacy) )); }
while true; do
  while [ "$(busy)" -ge "$MAXP" ]; do sleep 10; done
  next=$(ls "$D/pending" 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then
    if [ -f "$D/stop" ] && [ "$(busy)" -eq 0 ]; then echo "$(date -u +%TZ) QUEUE-EXIT" >> "$LOG"; break; fi
    sleep 20; continue
  fi
  mv "$D/pending/$next" "$D/running/$next"
  line=$(cat "$D/running/$next")
  echo "$(date -u +%TZ) START $next: $line" >> "$LOG"
  ( nice -n 19 ionice -c3 bash -c "$line" >> "$LOG" 2>&1 < /dev/null; rc=$?
    mv "$D/running/$next" "$D/done/$next"; echo "$(date -u +%TZ) END $next rc=$rc" >> "$LOG" ) &
  sleep 2
done
