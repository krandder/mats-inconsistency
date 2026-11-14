#!/bin/bash
# P1 third resume, after the spot preemption of trader-cat-p1-3 (16:31Z). /root/out = the p1-3 sync of 16:29Z:
# T2m complete (not rerun here); maze TT fit at 59 checkpointed after batch 6 of 16 (maze/heads_tt_n59.pt.partial).
# Mazes only: data for 59/99 again (fresh /root/data; new marker names), then msize59 (resumes the TT fit from the
# partial) and msize99. The maze resume test already passed on p1-3 (PREREG Addendum G), so there is no GO_MAZE wait.
# Name contains "run_vm.sh" for the watchdog.
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out; DATA=/root/data
CK=$DATA/model_best.pth
mkdir -p $OUT/markers $OUT/logs $OUT/maze $DATA
touch $OUT/JOB_RUNNING
rm -f $OUT/JOB_DONE
( while true; do touch $OUT/JOB_RUNNING; sleep 60; done ) &
KEEP=$!
ts() { date -u +%FT%TZ; }
step() {  # step NAME cmd...
  local name=$1; shift
  [[ -f $OUT/markers/$name.done ]] && return 0
  echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"
  else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi
}
get_maze() {
  local n=$1 d=$DATA/maze_data_test_$1
  [[ -f $d/solutions.npy ]] && return 0
  curl -sSL --retry 3 -o $DATA/t$n.tgz https://www.cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_test_$n.tar.gz &&
    tar -C $DATA -xzf $DATA/t$n.tgz && rm -f $DATA/t$n.tgz && ls $d
}
sha256sum $CK $DATA/pilot_Bds.pt >$OUT/inputs4.sha256
sha256sum /root/job/*.py /root/job/*.sh >$OUT/code4.sha256
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu4.txt

maze_chain() {
  local M="--out $OUT/maze --data $DATA --ckpt $CK"
  export P1_GPU_FRAC=0.6
  step dl59d get_maze 59 || return 1
  step gen99d $PY -c "from data import write_generated; write_generated('$DATA', 99, 350, seed0=99000, procs=2, chunk=10)" &
  local G=$!
  step msize59 $PY eval.py maze-size --size 59 --splits dev,eval $M
  wait $G
  [[ -f $OUT/markers/gen99d.done ]] || return 1
  step msize99 $PY eval.py maze-size --size 99 --splits dev,eval $M
}

maze_chain &
MPID=$!
wait $MPID
kill $KEEP 2>/dev/null
echo "$(ts) resume4 maze chain finished"
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
