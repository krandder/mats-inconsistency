#!/bin/bash
# launch-retry.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR] [MAX_TRIES]
# Retries launch.sh on stockouts (every 3 min), zeroing each failed create's phantom reservation
# (audit-failed.sh). L4 only (the wm2 guard prices every VM at the L4 rate).
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-6}; TRIES=${5:-20}
G=/home/agent/workspaces/eurisko-study/li-net/wm2/gpu
L=/tmp/wm2/logs/launch-$RUN.log
for i in $(seq 1 $TRIES); do
  $G/launch.sh $RUN $MIN $JOBFILE $PAR
  rc=$?
  att=$(awk -v r="LAUNCH $RUN " 'index($0, r){buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $L)
  if [[ $rc != 0 ]] && ! grep -q "READY:" <<<"$att" && grep -q "gcloud.compute.instances.create\|does not have enough resources\|stockout" <<<"$att"; then
    $G/audit-failed.sh >> $L 2>&1
    echo "$(date -u +%FT%TZ) RETRY $RUN after stockout (try $i/$TRIES)" | tee -a $L
    sleep 180
    continue
  fi
  exit $rc
done
echo "$(date -u +%FT%TZ) GIVEUP $RUN after $TRIES stockouts" | tee -a $L
exit 9
