#!/bin/bash
# wave.sh WAVE MINUTES PAR [FALLBACK_A100_AFTER]: one rental per jobs/WAVE-vm*.txt (run id smallnet-WAVE-vmI),
# each through launch-retry.sh (stockout retries), started 40 s apart, all detached. Logs: /tmp/smallnet/logs/.
set -u
W=$1; MIN=$2; PAR=$3; FB=${4:-0}
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
W_LC=$(echo "$W" | tr 'A-Z' 'a-z')
for f in $S/jobs/$W-vm*.txt; do
  i=$(basename $f .txt | sed 's/.*-vm//')
  RUN=smallnet-$W_LC-vm$i
  setsid nohup $S/gpu/launch-retry.sh $RUN $MIN $f $PAR 30 $FB > /tmp/smallnet/logs/launch-retry-$RUN.out 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) started $RUN ($f, $(grep -c -- --arm $f) jobs)"
  sleep 40
done
