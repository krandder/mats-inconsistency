#!/bin/bash
# Minimal GCE startup script for soup-gpu VMs (opt-in: GPU_STARTUP=minimal in gpu-up.sh; added 2026-10-11 for wm2/eurisko).
# For Python/PyTorch jobs that need no clang, bun or Bend2: waits until the NVIDIA driver answers, makes sure rsync
# and curl exist, then writes /var/lib/soup/READY (or /var/lib/soup/FAILED). gpu-up.sh waits for one of them.
exec >>/var/log/soup-startup.log 2>&1
set -x
mkdir -p /var/lib/soup /root/runs
[[ -f /var/lib/soup/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/soup/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive
if ! command -v rsync >/dev/null || ! command -v curl >/dev/null; then
  for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
  apt-get install -y -q rsync curl || fail apt
fi
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi >/dev/null 2>&1 || fail nvidia-smi
echo 'export PATH=/usr/local/cuda/bin:$PATH' >/etc/profile.d/soup.sh
{
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)"
  echo "startup=minimal"
} >/var/lib/soup/READY.tmp && mv /var/lib/soup/READY.tmp /var/lib/soup/READY
