#!/bin/bash
# launch3-retry.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR] [MAX_TRIES]
# Retries launch.sh when the VM could not be created (stockout / quota), every 2 min. Any other failure is final.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-2}; TRIES=${5:-10}
G=/home/agent/workspaces/eurisko-study/li-net/eca/gpu
L=/tmp/eca/logs/launch-$RUN.log
for i in $(seq 1 $TRIES); do
  $G/launch3.sh $RUN $MIN $JOBFILE $PAR
  rc=$?
  att=$(awk -v r="LAUNCH $RUN " 'index($0, r){buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $L)
  if [[ $rc != 0 ]] && ! grep -q "READY:" <<<"$att" && grep -qiE "resources|stockout|QUOTA|CREATE-FAILED|gcloud.compute.instances.create" <<<"$att"; then
    echo "$(date -u +%FT%TZ) RETRY $RUN after create failure (try $i/$TRIES)" | tee -a $L
    sleep 120
    continue
  fi
  exit $rc
done
echo "$(date -u +%FT%TZ) GIVEUP $RUN after $TRIES create failures" | tee -a $L
exit 9
