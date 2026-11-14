#!/bin/bash
# launch4.sh NAME SEED [relaunch|new] [p3b|abl|reg] : create a phase-3b VM (or, with abl/reg, a DELETE-termination
# replacement for a phase-3 VM that could not restart) (DELETE termination, max-run 4 h), push code, the phase-2
# economy checkpoint (for halt64) and the jobs, start run_p3.sh. Kind "re" (21:40 UTC): a re-evaluation-only VM that
# takes jobs moved off a busy VM; its job files and the checkpoints to push are in results/re5/NAME/. With "relaunch" (after a spot preemption deleted
# the VM) the synced results/vm/NAME/p3 (checkpoints, .done markers) is pushed back first, so the runner resumes.
N=$1; s=$2; R=$3; K=${4:-p3b}; D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet; cd $D
out=$(./mkvm4.sh $N) || { echo "CREATEFAIL $N"; exit 1; }; echo "$out"; Z=$(echo $out | awk '{print $2}')
( flock 9; touch results/p4vms.txt; grep -v "^$N " results/p4vms.txt > results/p4vms.$N.tmp; echo "$N $Z $s $K" >> results/p4vms.$N.tmp; mv results/p4vms.$N.tmp results/p4vms.txt ) 9> /tmp/mrtest/p4vms.lock
timeout 300 bash -c "until gcloud compute ssh $N --zone $Z --project GCP_PROJECT --command true >/dev/null 2>&1; do sleep 15; done" || { echo "GSSHFAIL $N"; exit 1; }
IP=$(gcloud compute instances describe $N --zone $Z --project GCP_PROJECT --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
timeout 400 bash -c "until /tmp/mrtest/vmssh.sh $IP true 2>/dev/null; do sleep 10; done" || { echo "SSHFAIL $N"; exit 1; }
SSH="ssh -i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
/tmp/mrtest/vmssh.sh $IP 'mkdir -p p3 old'
if [ "$R" = relaunch ] && [ -d results/vm/$N/p3 ]; then
  rsync -a --exclude '*.tmp' --exclude p3.status -e "$SSH" results/vm/$N/p3/ user@$IP:p3/ || exit 1; fi
J=/tmp/mrtest/jobs_$N.$(date +%s); mkdir -p $J
case $K in abl) NEQ=3090 ./mkjobs3.sh $s $J; : > $J/solo.txt;; reg) ./mkregjobs3.sh $s $J; : > $J/solo.txt;;
  re) cp results/re5/$N/jobs.txt results/re5/$N/reeval.txt results/re5/$N/solo.txt $J/ || exit 1;; *) ./mkjobs4.sh $s $J;; esac
rsync -a -e "$SSH" econ2.py mresnet.py dtrecall.py reeval.py halt64.py memdepth4.py run_p3.sh clampwatch.py user@$IP:~/ || exit 1
/tmp/mrtest/vmssh.sh $IP 'touch reeval.py'   # clampwatch.py treats processes started after this as running the current code
rsync -a -e "$SSH" results/vm/trader-mresnet-4/p2/econ_cov_voc_s$s.pt results/vm/trader-mresnet-4/p2/econ_cov_voc_s$s.json results/vm/trader-mresnet-9/p2/bptt_cov_dscov_s$s.pt user@$IP:old/ || exit 1
rsync -a -e "$SSH" $J/jobs.txt $J/reeval.txt $J/solo.txt user@$IP:p3/ || exit 1
if [ "$K" = re ]; then rsync -a -e "$SSH" $(cat results/re5/$N/ckpts.txt) user@$IP:p3/ || exit 1; fi
/tmp/mrtest/vmssh.sh $IP 'touch p3/RUNNER; { pgrep -f "^bash run_p3.sh" >/dev/null || nohup bash run_p3.sh >> p3/runner.log 2>&1; } > /dev/null 2>&1 < /dev/null & sleep 3; ls p3 | tr "\n" " "; nvidia-smi --query-gpu=name --format=csv,noheader'
echo "LAUNCHED $N $Z seed $s ${R:-new} $K"
