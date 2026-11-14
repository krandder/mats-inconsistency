#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
for spec in "mlp mlp_res_L64_s0_sig0.0" "rnn rnn_T64_s0_sig0.0" "mlp mlp_res_L64_s0_sig0.01" "rnn rnn_T64_s1_sig0.01"; do
  set -- $spec
  nice -n 19 ionice -c3 $PY scratch/p5_rbvar.py $1 train/gpu_out/$2.npz runs/p5/rb_$2.json runs/p5/rbvar_$2.json
done
touch logs/q_p5e.done
