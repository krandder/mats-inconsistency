#!/usr/bin/env bash
# CPU queue (fourth agent): run the commands in CMDFILE (one per line, run from cm-interp-B) in order, at most MAXP at once, each at
# the lowest CPU and I/O priority with OMP_NUM_THREADS=1. Log: LOG (appended). usage: cpuq.sh CMDFILE LOG [MAXP]
cd /home/agent/workspaces/mats/work/cm-interp-B
CMDS=$1; LOG=$2; MAXP=${3:-4}
export OMP_NUM_THREADS=1
while read -r line; do
  [ -z "$line" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
  echo "$(date -u +%TZ) START $line" >> "$LOG"
  nice -n 19 ionice -c3 bash -c "$line" >> "$LOG" 2>&1 < /dev/null &
  sleep 1
done < "$CMDS"
wait
echo "$(date -u +%TZ) QUEUE-DONE $CMDS" >> "$LOG"
