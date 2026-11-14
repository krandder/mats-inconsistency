#!/usr/bin/env bash
# PREREG Amendment 2, tests T5a, T5b and T6, on a fourth L4. They were part of gpu_job2.sh, but the second job did not
# stage amp.py / lyap.py / comp.py (staging bug in launch_gpu.sh, fixed), so they failed there without running.
# Commands identical to gpu_job2.sh. Run from the staged lane dir; outputs under /root/runs/$1/.
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
mkdir -p $O
exec > >(tee -a $O/job.log) 2>&1
set -x
nvidia-smi -L; $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sha256sum lib/*.py scratch/*.py scratch/*.sh lib/rtable_ng.json
date -u
# T5a, T5b
$PY scratch/amp.py --n 128 --L 32 --nets 113,114,115,116 --cuts 16,24 --out $O/a2 || echo "T5a-128 FAILED"
$PY scratch/amp.py --n 256 --L 64 --nets 113,114,115,116 --cuts 32,48 --out $O/a2 || echo "T5a-256 FAILED"
$PY scratch/amp.py --n 512 --L 128 --nets 113,114,115,116 --cuts 64,96 --out $O/a2 || echo "T5a-512 FAILED"
date -u
$PY scratch/lyap.py --n 128 --L 32 --nets 113,114 --cuts 8 --Lps 4,8,16 --out $O/a2 || echo "T5b-128 FAILED"
$PY scratch/lyap.py --n 256 --L 64 --nets 113,114 --cuts 16 --Lps 8,16,32 --out $O/a2 || echo "T5b-256 FAILED"
$PY scratch/lyap.py --n 512 --L 128 --nets 113,114 --cuts 32 --Lps 16,32,64 --out $O/a2 || echo "T5b-512 FAILED"
date -u
# T6 compressor
$PY scratch/comp.py --n 128 --L 32 --net 113 --cut 24 --ks 1,4,16,64 --models pca,lin,mlp --out $O/a2 || echo "T6-128 FAILED"
$PY scratch/comp.py --n 256 --L 64 --net 113 --cut 48 --ks 1,4,16,64 --models pca,lin,mlp --out $O/a2 || echo "T6-256 FAILED"
$PY scratch/comp.py --n 512 --L 128 --net 113 --cut 96 --ks 1,4,16,64 --models pca,lin,mlp --out $O/a2 || echo "T6-512 FAILED"
$PY scratch/comp.py --n 256 --L 64 --net 113 --cut 32 --ks 1,4,16,64 --models pca,lin,mlp --out $O/a2 || echo "T6-256b FAILED"
date -u
echo JOB-DONE
