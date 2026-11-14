#!/bin/bash
# push.sh IP [OVERLAY_DIR]: push code, inputs, the p1-3 results and (optionally) a fresher pull overlay to a new P1 VM,
# clear the maze-data markers (fresh /root/data), verify sha256 of code, inputs and maze checkpoints on both sides.
set -u
IP=$1; OV=${2:-}
P1=/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P1; MR=$P1/../../market-resnet
CK=/tmp/p1ck/deep-thinking-drive/training-rusty-Tayla/model_best.pth
K=[ssh-key-path-removed]
E="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15"
R="rsync -a --rsync-path=sudo\ rsync"
cd $P1 || exit 1
timeout 60 bash /tmp/p1scratch/vm.sh 'mkdir -p /root/job /root/out /root/data' || exit 1
timeout 300 rsync -a --rsync-path="sudo rsync" -e "$E" ./*.py ./*.sh $MR/mresnet.py p1@$IP:/root/job/ || exit 1
timeout 300 rsync -a --rsync-path="sudo rsync" -e "$E" $MR/results/pilot/pilot_Bds.pt $CK p1@$IP:/root/data/ || exit 1
timeout 600 rsync -a --rsync-path="sudo rsync" -e "$E" --exclude 'SHA256SUMS*' --exclude FAILED_STEPS --exclude 'JOB_*' \
  --exclude '*.tmp' results/vm/trader-cat-p1-3/ p1@$IP:/root/out/ || exit 1
if [ -n "$OV" ]; then
  timeout 600 rsync -a --rsync-path="sudo rsync" -e "$E" --exclude '*.tmp' "$OV/" p1@$IP:/root/out/ || exit 1
fi
timeout 60 bash /tmp/p1scratch/vm.sh 'rm -f /root/out/markers/dl59*.done /root/out/markers/gen99*.done /root/out/FAILED_STEPS'
FILES="eval.py train_heads.py data.py resume4_run_vm.sh"
MZ=${P1_VERIFY:-"heads_tt_n59.pt.partial heads_maze_base.pt"}
L=$( (sha256sum $FILES $MR/mresnet.py $CK $MR/results/pilot/pilot_Bds.pt | awk '{print $1}'
      for f in $MZ; do
        if [ -n "$OV" ] && [ -f "$OV/maze/$f" ]; then sha256sum "$OV/maze/$f"; else sha256sum results/vm/trader-cat-p1-3/maze/$f; fi
      done | awk '{print $1}') )
V=$(timeout 60 bash /tmp/p1scratch/vm.sh "cd /root/job; sha256sum $FILES mresnet.py /root/data/model_best.pth /root/data/pilot_Bds.pt $(for f in $MZ; do printf '/root/out/maze/%s ' $f; done)" | awk '{print $1}')
if [ "$L" == "$V" ]; then echo "PUSHED sha256 match ($(echo "$L" | wc -l) files)"; else echo "SHA MISMATCH"; diff <(echo "$L") <(echo "$V"); exit 2; fi
