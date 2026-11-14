#!/bin/bash
# GCE startup script for trader-cat-p7-* CPU VMs: python venv with CPU torch, then /var/lib/p7/READY (or FAILED).
exec >>/var/log/p7-startup.log 2>&1
set -x
mkdir -p /var/lib/p7 /root/out /root/job
[[ -f /var/lib/p7/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/p7/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive HOME=/root
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q python3-venv python3-pip rsync || fail apt
python3 -m venv /opt/venv || fail venv
/opt/venv/bin/pip install -q --upgrade pip || fail pip
/opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || fail pip-torch
/opt/venv/bin/pip install -q numpy || fail pip-numpy
/opt/venv/bin/python -c "import torch, numpy; print(torch.__version__)" || fail import
{
  echo "cpus=$(nproc)"
  echo "torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"
} >/var/lib/p7/READY.tmp && mv /var/lib/p7/READY.tmp /var/lib/p7/READY
