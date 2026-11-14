#!/bin/bash
# Follow-up queue: locate then split per net (one process per net pair), RB tails in a third process.
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
OUTD=train/gpu_out
run_net() {
  kind=$1; b=$2
  $N9 $PY scratch/p5_locate.py $kind $OUTD/$b.npz runs/p5/locate_$b.json 20 >> logs/p5_locate.log 2>&1
  $N9 $PY scratch/p5_split.py $kind $OUTD/$b.npz runs/p5/locate_$b.json runs/p5/split_$b.json >> logs/p5_split.log 2>&1
}
case $1 in
  a) run_net mlp mlp_res_L64_s0_sig0.0; run_net rnn rnn_T64_s0_sig0.0 ;;
  b) run_net rnn rnn_T64_s1_sig0.01; run_net mlp mlp_res_L64_s0_sig0.01 ;;
  c) for spec in "mlp mlp_res_L64_s0_sig0.0" "rnn rnn_T64_s0_sig0.0" "mlp mlp_res_L64_s0_sig0.01" "rnn rnn_T64_s1_sig0.01"; do
       set -- $spec; $N9 $PY scratch/p5_rb.py $1 $OUTD/$2.npz runs/p5/rb_$2.json 16 >> logs/p5_rb.log 2>&1; done ;;
esac
echo DONE > logs/q_p5_$1.done
