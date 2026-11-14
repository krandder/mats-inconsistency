#!/bin/bash
# Runs on a cdistill VM (root): train/eval each arm for one student, sequentially.
# Usage: distill_remote.sh MODEL "ARM ARM ..." [extra distill_student.py flags...]
set -u
MODEL=$1; ARMS=$2; shift 2
cd /root/job
mkdir -p /root/out; rm -f /root/out/JOB_DONE /root/out/JOB_FAILED
touch /root/out/JOB_RUNNING
rc=0
for arm in $ARMS; do
  out=/root/out/$MODEL/$arm
  if [[ -f $out/result.json ]]; then echo "skip $MODEL $arm (done)"; continue; fi
  /opt/venv/bin/python scripts/distill_student.py --bundle /root/job/bundle --model /root/models/$MODEL \
    --arm "$arm" --out "$out" "$@" >"/root/out/$MODEL-$arm.log" 2>&1 || { echo "arm $arm failed" >>/root/out/JOB_FAILED; rc=1; }
done
cp /root/models/$MODEL/REVISION /root/out/$MODEL/REVISION 2>/dev/null
rm -f /root/out/JOB_RUNNING
echo "rc=$rc $(date -u +%FT%TZ)" >/root/out/JOB_DONE
