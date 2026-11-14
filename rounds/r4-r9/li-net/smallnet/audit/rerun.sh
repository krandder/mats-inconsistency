#!/bin/bash
# Audit rerun (CPU only, no GPU) with the repo's own smallnet.py and the final protocol (es_boards=3000, patience 25, --test):
#   1. k0    at N=300 on FRESH training boards (train seed 5 = env seeds 1,500,000..; never used by the loop or the final runs)
#   2. rbtd8 at N=300 on the same fresh boards; MAXMIN caps training with the repo's --max-minutes (0 = no cap). The audit
#      used MAXMIN=25 (box at load 20-60): it stopped at step 460 (best 420), the reported runs ran 1220-2060 steps,
#      so the audit's rbtd8 number is a truncated, pessimistic one. A full run needs ~1 h at ~2-3 s/step.
#   3. control: k0 at N=300 on the reported seed-3 boards, to compare CPU with the reported GPU run (3.19e-3 TEST KL).
# Outputs: /tmp/smallnet-audit/runs/AUDIT-*.{json,log}. Torch (CPU build) from /tmp/smallnet/pylib. 3 processes, 5 threads.
set -u
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
D=/tmp/smallnet-audit/data; OUT=/tmp/smallnet-audit/runs
mkdir -p $D $OUT
# val/test caches verified identical to a fresh regeneration (audit/datacheck.py); seed-5 boards are generated on first use
cp --update=none /tmp/smallnet/data/val-n3000.post.npz /tmp/smallnet/data/test-n5000.post.npz /tmp/smallnet/data/train-n10000-s3.npz $D/
export PYTHONPATH=/tmp/smallnet/pylib LIR_HARNESS=/home/agent/workspaces/eurisko-study/lir-agent/harness \
  SMALLNET_DATA=$D PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python3 -c "import sys; sys.path.insert(0, '$S'); import data; data.load_split('train', 10000, seed=5)"   # fresh boards, once
OMP_NUM_THREADS=1 python3 $S/smallnet.py --arm k0 --n 300 --seed 5 --tag AUDIT-k0 --set es_boards=3000 --test --threads 1 \
  --device cpu --out $OUT/AUDIT-k0-n300-s5.json > $OUT/AUDIT-k0-n300-s5.log 2>&1 &
OMP_NUM_THREADS=3 python3 $S/smallnet.py --arm rbtd --n 300 --seed 5 --tag AUDIT-rbtd8 --set rbtd=8 es_boards=3000 --test --threads 3 \
  --device cpu --max-minutes ${MAXMIN:-0} --out $OUT/AUDIT-rbtd8-n300-s5.json > $OUT/AUDIT-rbtd8-n300-s5.log 2>&1 &
OMP_NUM_THREADS=1 python3 $S/smallnet.py --arm k0 --n 300 --seed 3 --tag AUDIT-k0-cpuctl --set es_boards=3000 --test --threads 1 \
  --device cpu --out $OUT/AUDIT-k0cpuctl-n300-s3.json > $OUT/AUDIT-k0cpuctl-n300-s3.log 2>&1 &
wait
