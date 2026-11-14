#!/usr/bin/env bash
# Queue of CPU analyses on fetched nets (lowest priority, at most 2 lanes here; E5 uses the third):
# lane 1: tokan.py (perm, M2, M3, M5, M6) on tok W0 N = 3, 12 (seeds 0, 1);
# lane 2: m1prof.py (M1 with deviation shares) on tok, set, setnc (N = 3, 12; seeds 0, 1) and tree16 (N = 3, 12).
cd /home/agent/workspaces/mats/work/scale-free-exp
export OMP_NUM_THREADS=1
P="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python"
B3=local/sfx-b3; B4=local/sfx-b4
wait_for() { until [ -f "$1/eval_test_model.json" ]; do sleep 60; ./fetch.sh > /dev/null 2>&1; done; }
(
  for N in 3 12; do for s in 0 1; do wait_for $B4/e1w0_tok_N${N}_s$s; done; done
  $P tokan.py $B4/e1w0_tok_N3_s0 $B4/e1w0_tok_N3_s1 $B4/e1w0_tok_N12_s0 $B4/e1w0_tok_N12_s1 --n 2000 --out results/tokan.jsonl
) > anq_lane1.log 2>&1 &
(
  for N in 3 12; do for s in 0 1; do wait_for $B4/e1w0_tok_N${N}_s$s; done; done
  $P m1prof.py $B4/e1w0_tok_N3_s0 $B4/e1w0_tok_N3_s1 $B4/e1w0_tok_N12_s0 $B4/e1w0_tok_N12_s1 --n 3000 --out results/m1.jsonl
  for N in 3 12; do for s in 0 1; do wait_for $B3/e1w0_set_N${N}_s$s; wait_for $B3/e1w0_setnc_N${N}_s$s; done; wait_for $B3/e1w0_tree16_N${N}_s0; done
  L=""; for N in 3 12; do for s in 0 1; do L="$L $B3/e1w0_set_N${N}_s$s $B3/e1w0_setnc_N${N}_s$s"; done; L="$L $B3/e1w0_tree16_N${N}_s0"; done
  $P m1prof.py $L --n 3000 --out results/m1.jsonl
) > anq_lane2.log 2>&1 &
wait
echo ANQ-DONE
