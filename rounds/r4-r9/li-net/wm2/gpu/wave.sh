#!/bin/bash
# wave.sh WAVE TASK MINUTES PAR: one rental per jobs/WAVE-TASK-vm*.txt (run id wm2-<wave>-<task>-vm<i>), each through
# launch-retry.sh (stockout retries), started 30 s apart, all detached. Logs: /tmp/wm2/logs/.
set -u
W=$1; TK=$2; MIN=$3; PAR=$4
D=/home/agent/workspaces/eurisko-study/li-net/wm2
W_LC=$(echo "$W" | tr 'A-Z' 'a-z')
for f in $D/jobs/$W-$TK-vm*.txt; do
  i=$(basename $f .txt | sed 's/.*-vm//')
  RUN=wm2-$W_LC-$TK-vm$i
  setsid nohup $D/gpu/launch-retry.sh $RUN $MIN $f $PAR 20 > /tmp/wm2/logs/launch-retry-$RUN.out 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) started $RUN ($f, $(grep -c -- --task $f) jobs)"
  sleep 30
done
