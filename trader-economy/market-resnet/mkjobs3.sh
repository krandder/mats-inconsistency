#!/bin/bash
# write p3/jobs.txt and p3/reeval.txt for seed $1 into directory $2 (NEQ = equal-FLOP iterations of the width-96 DT-recall)
s=$1; O=$2; NEQ=${NEQ:?}; mkdir -p $O
C="--iters 3000 --bs 64 --D 24 --evalL 32,320,3200 --seed $s"
B="--mode bptt --stream cov --ds 1 --covsup 1"
cat > $O/jobs.txt <<J
dtr_full_s$s dtr --iters 12000 --seed $s
econ_cov_voc_rr_s$s econ2 --mode econ --stream cov --voc 1 --train_pick roundrobin --auctionL none $C
econ_cov_s$s econ2 --mode econ --stream cov --auctionL none $C
bptt_cov_dscov_rand_s$s econ2 $B --train_pick random $C
dtr_eqtime_s$s dtr --time_budget 700 --seed $s
bptt_cov_dscov_tb1_rand_s$s econ2 $B --tbptt 1 --train_pick random $C
bptt_cov_dscov_tb1_s$s econ2 $B --tbptt 1 $C
bptt_cov_progcov_s$s econ2 --mode bptt --stream cov --prog 1 --covsup 1 $C
dtr_eqflop_s$s dtr --width 96 --iters $NEQ --seed $s
J
{ echo "re_old_econ_cov_voc_s$s reeval --ckpt old/econ_cov_voc_s$s.pt"
  echo "re_old_bptt_cov_dscov_s$s reeval --ckpt old/bptt_cov_dscov_s$s.pt"
  awk '{print "re_" $1 " reeval --ckpt p3/" $1 ".pt"}' $O/jobs.txt; } > $O/reeval.txt
