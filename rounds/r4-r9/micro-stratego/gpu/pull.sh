#!/bin/bash
# pull.sh RUNID : rsync /root/runs/ms/ from the VM into micro-stratego/runs/
RUN=$1
D=/home/agent/workspaces/mats/work/micro-stratego
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
read -r NAME ZONE IP < $D/gpu/vm-$RUN
RSH="ssh ${SSH_OPTS[*]}"
mkdir -p $D/runs
rsync -az --rsync-path='sudo rsync' -e "$RSH" $SSH_USER@$IP:/root/runs/ms/ $D/runs/
