#!/bin/bash
# launch3.sh NAME SEED abl|reg : create a p3 VM, push code, jobs (and the phase-2 checkpoints for abl), start run_p3.sh
N=$1; s=$2; K=$3; D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet; cd $D
if [ -n "$4" ]; then Z=$4; else   # $4 = zone of an existing VM (resume a launch)
  out=$(/tmp/mrtest/mkvm3.sh $N) || { echo "CREATEFAIL $N"; exit 1; }; echo "$out"; Z=$(echo $out | awk '{print $2}')
  echo "$N $Z" >> results/p3vms.txt; fi
# one gcloud ssh first: it installs the ssh key on the VM, after which plain ssh/rsync work
timeout 300 bash -c "until gcloud compute ssh $N --zone $Z --project GCP_PROJECT --command true >/dev/null 2>&1; do sleep 15; done" || { echo "GSSHFAIL $N"; exit 1; }
IP=$(gcloud compute instances describe $N --zone $Z --project GCP_PROJECT --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
timeout 400 bash -c "until /tmp/mrtest/vmssh.sh $IP true 2>/dev/null; do sleep 10; done" || { echo "SSHFAIL $N"; exit 1; }
SSH="ssh -i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
J=/tmp/mrtest/jobs_$N; rm -rf $J
if [ $K = abl ]; then NEQ=3090 ./mkjobs3.sh $s $J; else ./mkregjobs3.sh $s $J; fi
/tmp/mrtest/vmssh.sh $IP 'mkdir -p p3 old'
rsync -a -e "$SSH" econ2.py mresnet.py dtrecall.py reeval.py run_p3.sh user@$IP:~/ || exit 1
[ $K = abl ] && { rsync -a -e "$SSH" results/vm/trader-mresnet-4/p2/econ_cov_voc_s$s.pt results/vm/trader-mresnet-9/p2/bptt_cov_dscov_s$s.pt user@$IP:old/ || exit 1; }
rsync -a -e "$SSH" $J/jobs.txt $J/reeval.txt user@$IP:p3/ || exit 1
/tmp/mrtest/vmssh.sh $IP 'touch p3/RUNNER; pgrep -f "^bash run_p3.sh" >/dev/null || nohup bash run_p3.sh >> p3/runner.log 2>&1 < /dev/null & sleep 3; ls p3 | tr "\n" " "; nvidia-smi --query-gpu=name --format=csv,noheader'
echo "LAUNCHED $N $Z seed $s $K"
