#!/usr/bin/env bash
# PREREG Amendment 3 on a third L4 (run from the staged lane dir; outputs under /root/runs/$1/).
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID; T=$O/truth
mkdir -p $O
exec > >(tee -a $O/job.log) 2>&1
set -x
nvidia-smi -L; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sha256sum lib/*.py scratch/*.py scratch/*.sh lib/rtable_ng.json
date -u
$PY scratch/run2.py e1 --n 128 --L 32 --nets 113,114,115,116 --nt 24 --truthdir $T --out $O/a3 --cuts 16,24 \
  --fams EMP,PB8,PB16,PB32,PB64 --reps 1 --nfit 20 --nr 21 || echo "A3-128 FAILED"
date -u
$PY scratch/run2.py e1 --n 512 --L 128 --nets 113,114,115,116 --nt 23 --truthdir $T --out $O/a3 --cuts 64,96 \
  --fams EMP,PB32,PB64,PB128,PB256 --reps 1 --nfit 19 --nr 20 || echo "A3-512 FAILED"
date -u
echo JOB-DONE
