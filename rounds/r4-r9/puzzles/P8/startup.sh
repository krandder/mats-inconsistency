#!/bin/bash
# GCE startup script for trader-cat-p8-* VMs: python venv with torch (CUDA), then /var/lib/p8/READY (or FAILED).
exec >>/var/log/p8-startup.log 2>&1
set -x
mkdir -p /var/lib/p8 /root/out /root/job
[[ -f /var/lib/p8/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/p8/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive HOME=/root
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q python3-venv python3-pip rsync || fail apt
python3 -m venv /opt/venv || fail venv
/opt/venv/bin/pip install -q --upgrade pip || fail pip
/opt/venv/bin/pip install -q torch numpy scipy requests || fail pip-install
/opt/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || fail cuda
{
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
} >/var/lib/p8/READY.tmp && mv /var/lib/p8/READY.tmp /var/lib/p8/READY
