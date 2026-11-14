#!/bin/bash
# sh.sh RUNID CMD...  : run CMD as root on the VM (cwd /root/ms, PYTHONPATH=/root/pylib)
RUN=$1; shift
D=/home/agent/workspaces/mats/work/micro-stratego/gpu
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
read -r NAME ZONE IP < $D/vm-$RUN
printf 'cd /root/ms; export PYTHONPATH=/root/pylib PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2; %s\n' "$*" | vm_ssh "$IP" 'sudo bash -s'
