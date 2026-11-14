#!/bin/bash
# GCE startup script for tn-gpu VMs (root, at boot). Python venv /opt/tn with CUDA torch + numpy.
# Writes /var/lib/tn/READY (or /var/lib/tn/FAILED); gpu-up.sh waits for one of them.
exec >>/var/log/tn-startup.log 2>&1
set -x
mkdir -p /var/lib/tn /root/job /root/out
[[ -f /var/lib/tn/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/tn/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q python3-venv python3-pip rsync || fail apt
python3 -m venv /opt/tn || fail venv
/opt/tn/bin/pip install -q --upgrade pip || fail pip-upgrade
/opt/tn/bin/pip install -q torch numpy scipy pandas matplotlib || fail pip-install
echo 'export PATH=/opt/tn/bin:/usr/local/cuda/bin:$PATH' >>/etc/profile.d/tn.sh
/opt/tn/bin/python -c 'import torch; assert torch.cuda.is_available(), "no cuda"; print("torch="+torch.__version__, "cuda="+str(torch.version.cuda), "gpu="+torch.cuda.get_device_name(0).replace(" ","_"))' >>/var/lib/tn/READY.tmp || fail torch-cuda
echo "python=$(/opt/tn/bin/python --version | tr ' ' _) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" >>/var/lib/tn/READY.tmp
mv /var/lib/tn/READY.tmp /var/lib/tn/READY
