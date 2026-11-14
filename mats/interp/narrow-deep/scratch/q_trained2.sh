#!/bin/bash
# Part 4 pipeline: copy GPU outputs, evaluate, select, analyse all selected, estimate on a subset (time).
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
SRC=/tmp/l4runs/nd-train-2/runs/nd-train-2/out
while [ ! -f runs/p3/truth_res_n16_L256_s0.001_seed3_att.json ]; do sleep 5; done
mkdir -p train/gpu_out runs/p4
cp -n $SRC/*.npz $SRC/*.json $SRC/*.log train/gpu_out/ 2>/dev/null
$N9 $PY scratch/p4_eval.py runs/p4/eval_all.json train/gpu_out/*.npz > logs/p4_eval.log 2>&1
$N9 $PY scratch/p4_select.py > runs/p4/selected.txt 2>> logs/p4_eval.log
grep -E "rnn_T64|rnn_T32_s[0-9]_sig0.0\.|mlp_res_L64|mlp_plain_L16_s[0-9]_sig0.0\." runs/p4/selected.txt > runs/p4/selected_est.txt
while read kind path; do
  b=$(basename $path .npz)
  $N9 $PY scratch/p4_analyze.py $kind $path runs/p4/an_$b.json >> logs/p4_analyze.log 2>&1
done < runs/p4/selected.txt
while read kind path; do
  b=$(basename $path .npz)
  $N9 $PY scratch/p4_est.py $kind $path runs/p4/est_$b.json 16 >> logs/p4_est.log 2>&1
done < runs/p4/selected_est.txt
echo DONE > logs/q_trained.done
