#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
for spec in "mlp mlp_res_L64_s0_sig0.0" "rnn rnn_T64_s0_sig0.0"; do
  set -- $spec
  $N9 $PY scratch/p5_tailcheck.py $1 train/gpu_out/$2.npz runs/p5/locate_$2.json runs/p5/split_$2.json runs/p5/tailcheck_$2.json >> logs/p5_tailcheck.log 2>&1
done
while [ ! -f logs/q_p5_b.done ]; do sleep 20; done
for spec in "rnn rnn_T64_s1_sig0.01" "mlp mlp_res_L64_s0_sig0.01"; do
  set -- $spec
  $N9 $PY scratch/p5_tailcheck.py $1 train/gpu_out/$2.npz runs/p5/locate_$2.json runs/p5/split_$2.json runs/p5/tailcheck_$2.json >> logs/p5_tailcheck.log 2>&1
done
echo DONE > logs/q_p5_d.done
