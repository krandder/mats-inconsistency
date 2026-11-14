#!/bin/bash
# Runs on the VM: every line of the queue is "NAME shell command..."; P parallel slots (OMP threads 1 unless the
# command sets its own). Logs -> /root/out/logs/NAME.log; exit 0 -> /root/out/markers/NAME.done (skipped on restart),
# else NAME.failed. Markers JOB_RUNNING / JOB_DONE in /root/out. Several runners may share a VM.
cd /root/job || exit 1
P=${1:-30}
Q=${2:-queue.txt}
mkdir -p /root/out/logs /root/out/markers
rm -f /root/out/JOB_DONE
touch /root/out/JOB_RUNNING "/root/out/RUNNER.$$"
run_one_fu() {
  local name=$1; shift
  [[ -f /root/out/markers/$name.done ]] && return 0
  rm -f "/root/out/markers/$name.failed"
  if (cd /root/job && PY=/opt/venv/bin/python OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1 bash -c "$*") >>"/root/out/logs/$name.log" 2>&1; then
    date -u +%FT%TZ >"/root/out/markers/$name.done"
  else
    date -u +%FT%TZ >"/root/out/markers/$name.failed"
  fi
}
export -f run_one_fu
grep -v '^\s*$' "/root/job/$Q" | while IFS= read -r line; do printf '%s\0' "$line"; done | \
  xargs -0 -P "$P" -I{} bash -c 'set -- $1; n=$1; shift; run_one_fu "$n" "$*"' _ {}
rm -f "/root/out/RUNNER.$$"
if ! ls /root/out/RUNNER.* >/dev/null 2>&1; then
  rm -f /root/out/JOB_RUNNING
  date -u +%FT%TZ > /root/out/JOB_DONE
fi
