#!/bin/bash
# GCE startup script for trader-bxd-* VMs: venv with torch (CUDA), the EqR code at a pinned commit, the released
# Sudoku-Extreme checkpoint and test arrays (sha256-checked), then /var/lib/p17/READY (or FAILED).
exec >>/var/log/p17-startup.log 2>&1
set -x
mkdir -p /var/lib/p17 /root/out /root/job /root/eqr-art
[[ -f /var/lib/p17/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/p17/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive HOME=/root
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q python3-venv python3-pip rsync git || fail apt
python3 -m venv /opt/venv || fail venv
/opt/venv/bin/pip install -q --upgrade pip || fail pip
/opt/venv/bin/pip install -q torch numpy pydantic omegaconf colorama || fail pip-install
/opt/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || fail cuda
cd /root/eqr-art
git clone -q https://github.com/locuslab/EqR.git eqr && git -C eqr checkout -q 134c7541711125860954c78463023c583b2ef21b || fail git
HF=https://huggingface.co
get() { for i in 1 2 3 4 5 6; do curl -sfL -o "$2" "$1" && return 0; sleep 20; done; return 1; }
mkdir -p data/test
get $HF/locuslab/EqR-model/resolve/main/sudoku-extreme/eqr.pth eqr_sudoku.pth || fail ckpt
get $HF/datasets/locuslab/EqR-data/resolve/main/sudoku-extreme-1k-aug-1000/test/all__inputs.npy data/test/all__inputs.npy || fail data
get $HF/datasets/locuslab/EqR-data/resolve/main/sudoku-extreme-1k-aug-1000/test/all__labels.npy data/test/all__labels.npy || fail data
cat >SHA <<'EOF'
258b1f0a90003a2db0de0c8df7831683ffce00de295ed4bba15dc8e599920d26  eqr_sudoku.pth
cbc2ab9f1743d79da9c0c26273460140bf542a5fb3229cfdd80575615163bcd5  data/test/all__inputs.npy
3d0bb8e73ecc5681675803c1cb518ebb4e771dac93a6741256a7a85ab8e29c2b  data/test/all__labels.npy
EOF
sha256sum -c SHA || fail sha256
{
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
} >/var/lib/p17/READY.tmp && mv /var/lib/p17/READY.tmp /var/lib/p17/READY
