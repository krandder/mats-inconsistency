#!/bin/bash
# Runs on the VM: every line of queue.txt is "NAME p14.py-args..."; P parallel single-thread jobs.
# Results -> /root/out/runs/, logs -> /root/out/logs/NAME.log. Markers JOB_RUNNING / JOB_DONE in /root/out.
# p14.py skips runs whose JSON exists and resumes from its .partial checkpoint, so a restart is safe.
cd /root/job || exit 1
P=${1:-30}
Q=${2:-queue.txt}
mkdir -p /root/out/runs /root/out/logs
rm -f /root/out/JOB_DONE
touch /root/out/JOB_RUNNING "/root/out/RUNNER.$$"
run_one() {
  local name=$1; shift
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/venv/bin/python p14.py "$@" --out /root/out/runs >>"/root/out/logs/$name.log" 2>&1
}
export -f run_one
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ < "/root/job/$Q"
# several runners may share a VM: the last one to finish marks the VM idle
rm -f "/root/out/RUNNER.$$"
if ! ls /root/out/RUNNER.* >/dev/null 2>&1; then
  rm -f /root/out/JOB_RUNNING
  date -u +%FT%TZ > /root/out/JOB_DONE
fi
