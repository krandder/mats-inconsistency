#!/bin/bash
# down.sh RUNID : delete the run's VM (gpu-down.sh --only NAME --no-sync); results must be pulled first
RUN=$1
D=/home/agent/workspaces/mats/work/micro-stratego/gpu
read -r NAME ZONE IP < $D/vm-$RUN
/home/agent/workspaces/eurisko-study/logical-soup/infra/gpu-down.sh --only "$NAME" --no-sync --reason "r4-e1 $RUN done" 2>&1 | tail -2
mv $D/vm-$RUN $D/logs/vm-$RUN.deleted-$(date -u +%H%M)
