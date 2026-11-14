#!/bin/bash
# Run one queue job end to end: VM up -> copy job dir -> run.sh -> sync results -> VM down -> ledger.
# Usage: infra/job-run.sh JOBID      (started by queue-runner.sh; the job must be in queue/running/)
# The VM is deleted on every exit path (trap), and GCE deletes it anyway at its max-run-duration.
set -u
source "$(dirname "$0")/lib.sh"
JOB=${1:?JOBID}
valid_id "$JOB" || { echo "invalid job id" >&2; exit 2; }
D="$QUEUE/running/$JOB"
[[ -d $D && -f $D/run.sh && -f $D/job.env ]] || { echo "no runnable job at $D" >&2; exit 2; }
exec >>"$D/job.log" 2>&1
echo "$$" >"$D/runner.pid"
getv() { sed -n "s/^$1=//p" "$D/job.env" | tail -1 | tr -d '"'"'"; }
WS=$(getv WS); HOURS=$(getv HOURS); RESULTS_TO=$(getv RESULTS_TO); EXTRA=$(getv SYNC_INCLUDE)
DEFAULT_RTO=0; [[ -n $RESULTS_TO ]] || { RESULTS_TO=$D/out; DEFAULT_RTO=1; }
STARTED=$(now_utc); NAME=$(new_vm_name "$JOB"); echo "$NAME" >"$D/vm"
RUN_MIN=$(awk -v h="$HOURS" 'BEGIN{print int(h*60+0.999)}')
SCRATCH=/tmp/traders-net/G1/jobs/$JOB; mkdir -p "$SCRATCH" "$RESULTS_TO"
FINAL=failed; WHY="runner exited early"; RUN_RC=""
echo "$(now_utc) job $JOB ws=$WS hours=$HOURS vm=$NAME results_to=$RESULTS_TO"

status_json() {
  python3 -c 'import json,sys; k=sys.argv[1::2]; v=sys.argv[2::2]; print(json.dumps(dict(zip(k,v)),indent=2))' \
    job "$JOB" ws "$WS" vm "$NAME" state "$FINAL" why "$WHY" run_rc "$RUN_RC" hours "$HOURS" \
    started "$STARTED" ended "$(now_utc)" results_to "$RESULTS_TO" >"$D/status.json"
}
finish() {
  trap - EXIT INT TERM HUP
  [[ -n ${IP:-} ]] && sync_out
  if grep -q "vm=$NAME " "$GPU_LOG" 2>/dev/null; then
    "$INFRA/gpu-down.sh" --only "$NAME" --reason "job $JOB finished: $FINAL ($WHY)" || echo "WARNING: gpu-down failed; watchdog will retry"
  fi
  [[ $DEFAULT_RTO == 1 ]] && RESULTS_TO="$QUEUE/$FINAL/$JOB/out"
  status_json
  rm -f "$D/runner.pid"
  mv "$D" "$QUEUE/$FINAL/$JOB" 2>/dev/null || mv "$D" "$QUEUE/$FINAL/$JOB-$(date +%s)"
  echo "$(now_utc) job $JOB -> $FINAL ($WHY)"
  python3 "$INFRA/cost-report.py" >/dev/null 2>&1 || true
  exit 0
}
trap finish EXIT
trap 'WHY="runner signalled"; exit 130' INT TERM HUP

sync_out() {  # results only: csv/json/png/md/txt/log (+ SYNC_INCLUDE), files <= 100 MB
  local inc=(--include='*/' --include='*.csv' --include='*.json' --include='*.png' --include='*.md' --include='*.txt' --include='*.log')
  local p; for p in $EXTRA; do inc+=(--include="$p"); done
  rsync -a --prune-empty-dirs --max-size=100m --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" \
    "${inc[@]}" --exclude='*' "$SSH_USER@$IP:/root/out/" "$RESULTS_TO/" || echo "$(now_utc) sync failed"
}

requeue_preempted() {  # spot preemption: requeue at most twice per job, then fail
  local n; n=$(( $(cat "$D/preemptions" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$D/preemptions"
  if (( n <= 2 )); then FINAL=pending; WHY="$1; requeued (preemption $n of 2)"; else FINAL=failed; WHY="$1; 3rd preemption, not requeued"; fi
  exit 0
}

out=$("$INFRA/gpu-up.sh" --job "$JOB" --ws "$WS" --hours "$HOURS" --name "$NAME"); rc=$?
case $rc in
  0) ;;
  75) FINAL=pending; WHY="VM slots full at admission; requeued"; exit 0 ;;
  3) FINAL=refused; WHY="budget: $(tail -1 "$D/job.log")"; exit 0 ;;
  4) FINAL=pending; WHY="no L4 capacity in any zone; requeued"; sleep 300; exit 0 ;;
  7) requeue_preempted "preempted during startup" ;;
  *) WHY="gpu-up rc=$rc"; exit 1 ;;
esac
read -r _ ZONE IP <<<"$(tail -1 <<<"$out")"
echo "$(now_utc) VM ready $NAME $ZONE $IP"

cat >"$SCRATCH/tn-wrap.sh" <<EOF
#!/bin/bash
mkdir -p /root/out
cd /root/job
export OUT=/root/out PATH=/opt/tn/bin:/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1
nvidia-smi -L >/root/out/gpu.txt 2>&1
timeout --kill-after=60 ${RUN_MIN}m bash run.sh >/root/out/run.log 2>&1
echo \$? >/root/out/EXIT.tmp && mv /root/out/EXIT.tmp /root/out/EXIT
EOF
rsync -a --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" \
  --exclude=/out --exclude=/job.log --exclude=/runner.pid --exclude=/status.json \
  "$D/" "$SSH_USER@$IP:/root/job/" || { WHY="upload failed"; exit 1; }
rsync -a --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$SCRATCH/tn-wrap.sh" "$SSH_USER@$IP:/root/tn-wrap.sh" || { WHY="upload failed"; exit 1; }
vm_ssh "$IP" 'sudo setsid nohup bash /root/tn-wrap.sh </dev/null >/dev/null 2>&1 &' || { WHY="launch failed"; exit 1; }
log_gpu "RUN vm=$NAME ws=$WS job=$JOB run_timeout_min=$RUN_MIN"

deadline=$(( $(date +%s) + RUN_MIN*60 + 300 )); fails=0; i=0
while :; do
  sleep 60; i=$((i+1))
  e=$(vm_ssh "$IP" 'sudo cat /root/out/EXIT 2>/dev/null || echo RUNNING'); src=$?
  if [[ $src != 0 ]]; then
    fails=$((fails+1))
    if (( fails >= 5 )) && ! tn_vms | grep -q "^$NAME "; then IP=""; requeue_preempted "VM vanished mid-run (spot preemption or max-run-duration)"; fi
    continue
  fi
  fails=0
  [[ $e == RUNNING ]] || { RUN_RC=$e; break; }
  (( i % 5 == 0 )) && sync_out
  (( $(date +%s) < deadline )) || { WHY="local deadline passed"; exit 1; }
done
log_gpu "RUN-END vm=$NAME ws=$WS job=$JOB rc=$RUN_RC"
if [[ $RUN_RC == 0 ]]; then FINAL=done; WHY="run.sh rc=0"; else WHY="run.sh rc=$RUN_RC (124 = timeout)"; fi
exit 0
