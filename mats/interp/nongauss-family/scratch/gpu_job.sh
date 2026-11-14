#!/usr/bin/env bash
# PREREG runs R1-R4 on one L4 (run from the staged lane dir on the VM; outputs under /root/runs/$1/).
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID; T=$O/truth
mkdir -p $O
exec > >(tee -a $O/job.log) 2>&1
set -x
nvidia-smi -L; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sha256sum lib/*.py scratch/*.py scratch/*.sh lib/rtable_ng.json
date -u
# R1: main single cuts, 256 x 64 (net 112 first: ATTX reference paths only)
$PY scratch/run.py e1 --n 256 --L 64 --nets 112,113,114,115,116 --nt 24 --truthdir $T --out $O/r1 --cuts 16,32,48 \
  --fams EMP,PEMP,G,PG,PI,PB16,PC3I --fams1 PCA3,I,PR,PB2,PB4,PB8,PB32,PC3G,PC3R,ATT8,ATT16,ATTX \
  --reps 2 --nfit 20 --nr 21 || echo "R1 FAILED"
date -u
# R2: composition, 256 x 64
$PY scratch/run.py e2 --n 256 --L 64 --nets 113,114 --nt 24 --truthdir $T --out $O/r2 --c0 32 --two 32,48 \
  --cfams MC,EMP,G,PG,PI,PB16,PC3I --tfams G,PI,PB16 --ncs 20,21 || echo "R2 FAILED"
date -u
# R3: fixed L/n = 1/4
$PY scratch/run.py e1 --n 128 --L 32 --nets 113,114,115,116 --nt 24 --truthdir $T --out $O/r3 --cuts 8,16,24 \
  --fams EMP,PEMP,G,PG,PI,PB16,PB32,PC3I --reps 2 --nfit 20 --nr 21 || echo "R3a FAILED"
date -u
$PY scratch/run.py e1 --n 512 --L 128 --nets 113,114,115,116 --nt 23 --truthdir $T --out $O/r3 --cuts 32,64,96 \
  --fams EMP,PEMP,G,PG,PI,PB16,PB32,PC3I --reps 1 --nfit 19 --nr 20 || echo "R3b FAILED"
date -u
# R4: linearised composition, 256 x 64
$PY scratch/run.py lin --n 256 --L 64 --nets 113,114 --nt 24 --truthdir $T --out $O/r4 --lin0 32 \
  --linfams EMP,G,PI,PB16 --nfit 20 --nrlin 22 || echo "R4 FAILED"
date -u
echo JOB-DONE
