#!/bin/bash
# write p3/jobs.txt and p3/reeval.txt for the belief-regularizer arms, seed $1, into directory $2
s=$1; O=$2; mkdir -p $O
C="--mode econ --stream cov --voc 1 --auctionL none --iters 3000 --bs 64 --D 24 --evalL 32,320,3200 --seed $s"
cat > $O/jobs.txt <<J
econ_cov_voc_liqt_s$s econ2 $C --liq t
econ_cov_voc_liqsqrt_s$s econ2 $C --liq sqrt
econ_cov_voc_floor_s$s econ2 $C --floor 0.01
econ_cov_voc_l2b_s$s econ2 $C --l2b 0.001
econ_cov_voc_bcap_s$s econ2 $C --bcap 10
econ_cov_voc_dbcap_s$s econ2 $C --dbcap 1
J
awk '{print "re_" $1 " reeval --ckpt p3/" $1 ".pt"}' $O/jobs.txt > $O/reeval.txt
