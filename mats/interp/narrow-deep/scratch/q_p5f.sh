#!/bin/bash
# oracle gate mixture, 4 nets, at most 3 at a time
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
run() { nice -n 19 ionice -c3 $PY scratch/p5_oracle.py $1 train/gpu_out/$2.npz runs/p5/locate_$2.json runs/p5/oracle_$2.json 20 > logs/p5_oracle_$2.log 2>&1; }
run mlp mlp_res_L64_s0_sig0.0 &
run rnn rnn_T64_s0_sig0.0 &
run rnn rnn_T64_s1_sig0.01 &
wait
run mlp mlp_res_L64_s0_sig0.01
touch logs/q_p5f.done
