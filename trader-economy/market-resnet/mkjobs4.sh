#!/bin/bash
# write p3/jobs.txt, p3/reeval.txt and p3/solo.txt for the phase-3b arms (the 09-29 audit's missing controls), seed $1, into dir $2
s=$1; O=$2; mkdir -p $O
C="--iters 3000 --bs 64 --D 24 --evalL 32,320,3200 --seed $s"
B="--mode bptt --stream cov --ds 1 --covsup 1"
cat > $O/jobs.txt <<J
bptt_cov_dscov_tb1_voc_rep_rand_s$s econ2 $B --vocsup 1 --tbptt 1 --train_pick random --ptopt 1 --bidsup 1 $C
bptt_cov_dscov_tb1_voc_rand_s$s econ2 $B --vocsup 1 --tbptt 1 --train_pick random $C
bptt_cov_dscov_voc_rand_s$s econ2 $B --vocsup 1 --train_pick random $C
bptt_cov_progcov_a05_s$s econ2 --mode bptt --stream cov --prog 1 --alpha 0.5 --covsup 1 $C
bptt_cov_progcov_a0_s$s econ2 --mode bptt --stream cov --prog 1 --alpha 0 --covsup 1 $C
econ_cov_online_s$s econ2 --mode econ --stream cov --online 1 --auctionL none $C
bptt_cov_dscov_tb1_voc_bcap_rand_s$s econ2 $B --vocsup 1 --tbptt 1 --train_pick random --bcap 10 $C
bptt_cov_dscov_bcap_s$s econ2 $B --bcap 10 $C
halt64_s$s py halt64.py --ckpt old/econ_cov_voc_s$s.pt --out p3/halt64_s$s.json
J
awk '$2 == "econ2" {print "re_" $1 " reeval --ckpt p3/" $1 ".pt"}' $O/jobs.txt > $O/reeval.txt
if [ "$s" = 0 ]; then echo "memdepth4 py memdepth4.py 24,240,2400,24000 p3/memdepth4.json" > $O/solo.txt; else : > $O/solo.txt; fi
