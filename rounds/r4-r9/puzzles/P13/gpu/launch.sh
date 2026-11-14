#!/bin/bash
# launch.sh SLOT : create the slot's next VM (SLOT, then SLOT-r1, SLOT-r2, ...), push code + frozen data + M-ref,
# the synced results/vm/SLOT/out (checkpoints, .done markers: resume after a preemption), gpu/inputs/SLOT (models
# to evaluate) and gpu/queues/SLOT.txt, then start run_p13.sh. Records "slot vm zone created" in gpu/vms.tsv.
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/gpu/common.sh
S=$1
exec 8> $TMPD/launch_$S.lock; flock -n 8 || { echo "LAUNCH_BUSY $S"; exit 2; }
[ -f $G/STOP ] && { echo "STOPPED"; exit 1; }
k=$(awk -v s=$S '$1==s' $G/vms.tsv 2>/dev/null | wc -l)
[ "$k" = 0 ] && N=$S || N=$S-r$k
out=$($G/mkvm.sh $N) || { ev "CREATEFAIL $S $N"; exit 1; }
Z=$(echo $out | awk '{print $2}')
echo -e "$S\t$N\t$Z\t$(date -u +%FT%TZ)" >> $G/vms.tsv
ev "CREATED $S $N $Z"
timeout 420 bash -c "until gcloud compute ssh $N --zone $Z $PROJ --command true >/dev/null 2>&1; do sleep 15; done" || { ev "GSSHFAIL $N"; exit 1; }
IP=$(vm_ip $N $Z)
timeout 400 bash -c "until $SSH $U@$IP true 2>/dev/null; do sleep 10; done" || { ev "SSHFAIL $N"; exit 1; }
# code bundle (rebuilt every launch so code fixes propagate)
B=$TMPD/bundle_$S; rm -rf $B; mkdir -p $B/frozen_p0 $B/frozen $B/mref $B/out $B/in
cp $D/{econ_maze,eval_maze,dtr_maze,dtnet,check,flops}.py $D/gpu/*.py $G/run_p13.sh $B/ 2>/dev/null
cp $D/frozen_p0/{maze_train9.npz,mazes_e2h.npz,manifest.json} $B/frozen_p0/
cp $D/frozen/{p13_eval.npz,manifest.json} $B/frozen/
cp $D/../P0/src/dt_net.pth $B/mref/
[ -f $G/queues/$S.txt ] && cp $G/queues/$S.txt $B/out/queue.txt
[ -f $G/slots/$S ] && cp $G/slots/$S $B/out/SLOTS
$SSH $U@$IP 'mkdir -p p13/out p13/in' || exit 1
rsync -a -e "$SSH" $B/ $U@$IP:p13/ || { ev "PUSHFAIL $N"; exit 1; }
if [ -d $D/results/vm/$S/out ]; then
  rsync -a --exclude '*.tmp' --exclude HEARTBEAT --exclude RUNNER_DONE --exclude queue.txt --exclude SLOTS -e "$SSH" \
    $D/results/vm/$S/out/ $U@$IP:p13/out/ || { ev "PUSHFAIL-state $N"; exit 1; }
fi
[ -d $G/inputs/$S ] && rsync -a -e "$SSH" $G/inputs/$S/ $U@$IP:p13/in/
$SSH $U@$IP 'for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done; nvidia-smi --query-gpu=name --format=csv,noheader; cd p13; rm -f out/RUNNER_DONE; { pgrep -fx "bash run_p13.sh" >/dev/null || nohup bash run_p13.sh >> out/runner.log 2>&1; } > /dev/null 2>&1 < /dev/null & sleep 3; pgrep -fx "bash run_p13.sh" >/dev/null && echo RUNNER_UP' > $TMPD/launch_$N.out 2>&1
grep -q RUNNER_UP $TMPD/launch_$N.out && ev "LAUNCHED $S $N $Z $IP gpu=$(head -1 $TMPD/launch_$N.out)" || { ev "RUNNERFAIL $N $(tr '\n' ' ' < $TMPD/launch_$N.out)"; exit 1; }
