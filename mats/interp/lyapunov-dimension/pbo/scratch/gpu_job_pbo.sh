#!/usr/bin/env bash
# lyapunov-dimension PBO addendum on one L4 (run from the staged dir; outputs under /root/runs/$1/).
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
mkdir -p $O
exec > >(tee -a $O/job.log) 2>&1
set -x
nvidia-smi -L; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sha256sum lib/*.py scratch/*.py scratch/*.sh lib/rtable_ng.json PREREG-PBO.md
date -u
$PY scratch/run3.py e1 --n 256 --L 64 --nets 113,114,115,116 --nt 24 --truthdir truth --out $O/ra --cuts 32,48 --fams EMP,PB16,PB32,PB48,PB64,PBW16,PBW32,PBO8,PBO16,PBO24,PBO32,PBO48 --reps 1 --nfit 20 --nr 21 --nsm 14 || echo "RA FAILED"
date -u
$PY scratch/run3.py e2 --n 256 --L 64 --nets 113,114 --nt 24 --truthdir truth --out $O/rb --c0 32 --cfams MC,EMP,PB32,PBO32,PB64,PBO64 --tfams "" --ncs 20,21 || echo "RB FAILED"
date -u
echo JOB-DONE
