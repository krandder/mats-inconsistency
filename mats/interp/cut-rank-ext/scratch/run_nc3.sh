#!/bin/bash
# NC3 (PREREG.md): identifiability Jacobians, sequential, nice 19 / ionice idle, 2 BLAS threads.
cd /home/agent/workspaces/mats/work/cut-rank-ext
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "nc3 start $(date -u +%H:%M:%SZ)" >> logs/nc3_queue.log
for cfg in "4 2 2" "4 3 2" "6 2 2" "6 3 2"; do
  set -- $cfg
  nice -n 19 ionice -c3 python3 scratch/ident_jac.py $1 $2 $3 --K 64 --N 65536 --fronts 3 > logs/nc3_n$1_c$2_d$3.log 2>&1
  echo "nc3 n$1 c$2 d$3 exit $? $(date -u +%H:%M:%SZ)" >> logs/nc3_queue.log
done
echo "nc3 done $(date -u +%H:%M:%SZ)" >> logs/nc3_queue.log
