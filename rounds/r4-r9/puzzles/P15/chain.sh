#!/bin/bash
# chain.sh run_vm.sh-follow PLAN : wait until every running run_vm.sh on this VM has exited, then run run_vm.sh PLAN.
# The first argument keeps "run_vm.sh" in this process's command line, so the gpu.sh watchdog counts the VM as busy
# in the gap between the two plans. Used to append steps to a plan that is already running.
cd /root/job || exit 1
while pgrep -f '^bash run_vm.sh' >/dev/null; do sleep 10; done
PLAN=$2 exec bash run_vm.sh
