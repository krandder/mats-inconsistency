#!/bin/bash
# GCE startup script for trader-fu-* CPU VMs: python venv with CPU torch + numpy + scipy, then /var/lib/fu/READY (or FAILED).
exec >>/var/log/fu-startup.log 2>&1
set -x
mkdir -p /var/lib/fu /root/out /root/job
[[ -f /var/lib/fu/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/fu/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive HOME=/root
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q python3-venv python3-pip rsync || fail apt
python3 -m venv /opt/venv || fail venv
/opt/venv/bin/pip install -q --upgrade pip || fail pip
/opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || fail pip-torch
/opt/venv/bin/pip install -q numpy scipy || fail pip-numpy
/opt/venv/bin/python -c "import torch, numpy, scipy; print(torch.__version__)" || fail import
{
  echo "cpus=$(nproc)"
  echo "torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"
} >/var/lib/fu/READY.tmp && mv /var/lib/fu/READY.tmp /var/lib/fu/READY
