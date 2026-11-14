#!/bin/bash
# launch.sh RUNID SEED STAGE   (STAGE = train | eval)  : start the stage in the background on the VM (nohup), logs in
# /root/runs/ms/s<SEED>/<stage>.log ; train = train_ms.py + eval d0 ; eval = eval d1,d2,d3,d4,d4u,d5
RUN=$1; SEED=$2; STAGE=$3
D=/home/agent/workspaces/mats/work/micro-stratego/gpu
O=/root/runs/ms/s$SEED
if [[ $STAGE == train ]]; then
  CMD="mkdir -p $O && (python3 train_ms.py --seed $SEED --out $O > $O/train.log 2>&1 && python3 eval_ms.py --seed $SEED --out $O --parts d0 > $O/eval_d0.log 2>&1; echo EXIT \$? > $O/train.done)"
else
  CMD="(python3 eval_ms.py --seed $SEED --out $O --parts ${PARTS:-d1,d2,d3,d4,d4u,d5} > $O/eval_rest.log 2>&1; echo EXIT \$? > $O/eval.done)"
fi
$D/sh.sh $RUN "nohup bash -c '$CMD' > /dev/null 2>&1 &"
echo "$(date -u +%FT%TZ) launched $RUN seed=$SEED stage=$STAGE"
