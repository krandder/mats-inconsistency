#!/usr/bin/env bash
# PREREG Amendment 4 on a fifth L4: composition of the large-block PB families (run from the staged lane dir;
# outputs under /root/runs/$1/).
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID; T=$O/truth
mkdir -p $O
exec > >(tee -a $O/job.log) 2>&1
set -x
nvidia-smi -L; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sha256sum lib/*.py scratch/*.py scratch/*.sh lib/rtable_ng.json
date -u
$PY scratch/run.py e2 --n 256 --L 64 --nets 113,114 --nt 24 --truthdir $T --out $O/a4 --c0 32 --two 32,48 \
  --cfams MC,EMP,PB64,PB128 --tfams PB64 --ncs 20,21 || echo "A4-256 FAILED"
date -u
$PY scratch/run.py e2 --n 128 --L 32 --nets 113,114 --nt 24 --truthdir $T --out $O/a4 --c0 16 --two 16,24 \
  --cfams MC,EMP,PB32,PB64 --tfams PB32 --ncs 20,21 || echo "A4-128 FAILED"
date -u
echo JOB-DONE
