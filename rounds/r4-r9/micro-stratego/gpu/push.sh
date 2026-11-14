#!/bin/bash
# push.sh RUNID [--data] : rsync code (and data/ with --data) to /root/ms on the VM
RUN=$1
D=/home/agent/workspaces/mats/work/micro-stratego
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
read -r NAME ZONE IP < $D/gpu/vm-$RUN
RSH="ssh ${SSH_OPTS[*]}"
FILES=""
for f in env.py msgame.py model_ms.py train_ms.py eval_ms.py agents_ms.py; do [[ -f $D/$f ]] && FILES="$FILES $D/$f"; done
rsync -az --rsync-path='sudo rsync' -e "$RSH" $FILES $SSH_USER@$IP:/root/ms/
[[ ${2:-} == --data ]] && rsync -az --rsync-path='sudo rsync' -e "$RSH" $D/data $SSH_USER@$IP:/root/ms/
echo pushed $RUN
