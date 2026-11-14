#!/bin/bash
# Runs on a cdistill-v2 VM (root): train/eval the queued student arms, NPAR at a time.
# Usage: NPAR=2 distill2_remote.sh MODEL "ARM@TEACHER[#TAG] ..." [extra distill_student.py flags...]
#   ARM Z|R|C|O|CO; TEACHER 32b|235b (bundle /root/job/bundle-TEACHER); TAG an optional run name.
#   Output /root/out/MODEL/ARM-tTEACHER[-TAG]/, log /root/out/MODEL-ARM-tTEACHER[-TAG].log.
# An item whose result.json exists is skipped; any other item resumes from its newest checkpoint
# (distill_student.py does this itself), so rerunning the same queue after a restore is the resume.
set -u
MODEL=$1; QUEUE=$2; shift 2
export NPAR=${NPAR:-2} MODEL EXTRA="$*" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/job || exit 1
mkdir -p /root/out/$MODEL; rm -f /root/out/JOB_DONE /root/out/JOB_FAILED
touch /root/out/JOB_RUNNING
run_one() {
  local item=$1 tag="" arm teacher
  [[ $item == *#* ]] && { tag=-${item#*#}; item=${item%%#*}; }
  arm=${item%@*}; teacher=${item#*@}
  local name=$arm-t$teacher$tag
  local out=/root/out/$MODEL/$name
  [[ -f $out/result.json ]] && { echo "skip $name"; return 0; }
  # shellcheck disable=SC2086
  /opt/venv/bin/python scripts/distill_student.py --bundle /root/job/bundle-$teacher --model /root/models/$MODEL \
    --arm "$arm" --out "$out" $EXTRA >>"/root/out/$MODEL-$name.log" 2>&1 \
    || echo "item $name failed rc=$? $(date -u +%FT%TZ)" >>/root/out/JOB_FAILED
}
export -f run_one
printf '%s\n' $QUEUE | xargs -P "$NPAR" -I{} bash -c 'run_one "{}"'
cp /root/models/$MODEL/REVISION /root/out/$MODEL/REVISION 2>/dev/null
rc=0; [[ -f /root/out/JOB_FAILED ]] && rc=1
rm -f /root/out/JOB_RUNNING
echo "rc=$rc $(date -u +%FT%TZ)" >/root/out/JOB_DONE
