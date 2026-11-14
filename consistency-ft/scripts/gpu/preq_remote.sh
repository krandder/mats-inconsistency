#!/bin/bash
# Runs on a cdistill-preq VM (root): the prequential queue for one student, NPAR jobs at a time.
# Usage: preq_remote.sh MODEL "ARM:SEED ARM:SEED ..." [extra preq_student.py flags...]
# First checks packed vs ordinary scoring on the GPU (preq_check_packing.py); stops if it fails.
# A job whose result.json exists is skipped; a rerun resumes a job from its last block checkpoint.
set -u
MODEL=$1; QUEUE=$2; shift 2
export NPAR=${NPAR:-2} MODEL
export EXTRA="$*"
cd /root/job/scripts
mkdir -p /root/out/$MODEL; rm -f /root/out/JOB_DONE /root/out/JOB_FAILED
touch /root/out/JOB_RUNNING
S=/root/job/bundle/stream.jsonl.gz
# gate: packed == ordinary in fp32 (max |logit diff| <= 2e-3). Informative: the same at the jobs' bf16
# autocast (both paths carry bf16 noise, ~0.1 logit) and preq_check_precision.py (each vs fp32).
# (rerun at every start: the code may have changed)
f=/root/out/check-packing-fp32.json
if true; then
  /opt/venv/bin/python preq_check_packing.py --stream $S --model /root/models/$MODEL --device cuda --pairs 4 --fp32 \
    >$f.tmp 2>/root/out/check-packing-fp32.err && mv $f.tmp $f \
    || { echo "packing check fp32 failed" >>/root/out/JOB_FAILED; rm -f /root/out/JOB_RUNNING; echo "rc=1 $(date -u +%FT%TZ)" >/root/out/JOB_DONE; exit 1; }
fi
/opt/venv/bin/python preq_check_packing.py --stream $S --model /root/models/$MODEL \
  --device cuda --pairs 4 --tol 1 >/root/out/check-packing-bf16.json 2>/root/out/check-packing-bf16.err
/opt/venv/bin/python preq_check_precision.py >/root/out/check-precision.json 2>/root/out/check-precision.err
run_one() {
  local arm=${1%%:*} seed=${1##*:}
  local out=/root/out/$MODEL/$arm-s$seed
  [[ -f $out/result.json ]] && { echo "skip $1"; return 0; }
  # shellcheck disable=SC2086
  /opt/venv/bin/python preq_student.py --stream /root/job/bundle/stream.jsonl.gz --model /root/models/$MODEL \
    --arm "$arm" --seed "$seed" --out "$out" $EXTRA >>"/root/out/$MODEL-$arm-s$seed.log" 2>&1 \
    || echo "job $1 failed" >>/root/out/JOB_FAILED
}
export -f run_one
printf '%s\n' $QUEUE | xargs -P "$NPAR" -I{} bash -c 'run_one {}'
cp /root/models/$MODEL/REVISION /root/out/$MODEL/REVISION 2>/dev/null
rc=0; [[ -f /root/out/JOB_FAILED ]] && rc=1
rm -f /root/out/JOB_RUNNING
echo "rc=$rc $(date -u +%FT%TZ)" >/root/out/JOB_DONE
