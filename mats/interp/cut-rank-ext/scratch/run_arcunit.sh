#!/bin/bash
# Amendment 1 reruns of ARC2/ARC3 on unit-scaled cumulants, sequential, 1 BLAS thread, nice 19 / ionice idle.
cd /home/agent/workspaces/mats/work/cut-rank-ext
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/agent/.venvs/arc-noisy/bin/python
for cfg in "8 64 8 --arc3" "8 4 4 --arc3" "8 8 8 --arc3" "16 8 8"; do
  set -- $cfg
  until [ -f runs/nl_n$1_c$2_d$3.npz ]; do sleep 15; done
  nice -n 19 ionice -c3 $PY scratch/arc_unit_rerun.py $cfg > logs/arcunit_n$1_c$2_d$3.log 2>&1
  echo "arcunit n$1 c$2 d$3 exit $? $(date -u +%H:%M:%SZ)" >> logs/arcunit_queue.log
done
